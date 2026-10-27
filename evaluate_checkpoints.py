import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ==============================================================================
# Standalone Checkpoint Evaluation Script
# Loads the trained PyTorch neural weights directly from checkpoints/
# Stress-tests all models under severe dynamic disturbances (sigma = 0.6 N*m)
# and large initial angle shocks (up to 0.45 rad / ~26 degrees).
# ==============================================================================

device = 'mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[Device] Evaluating checkpoints on device: {device}")

# 1. Physics Engine
class DynamicBalancingEnv:
    def __init__(self, m=1.0, l=1.0, g=9.81, dt=0.02, u_max=15.0):
        self.m, self.l, self.g, self.dt, self.u_max = m, l, g, dt, u_max
        self.I = m * l**2
        self.K = np.array([28.0, 7.0])

    def step(self, state, action):
        th, dth = state
        u = float(np.clip(action, -self.u_max, self.u_max))
        def deriv(s, act): return np.array([s[1], (self.g / self.l) * np.sin(s[0]) + act / self.I])
        k1 = deriv(state, u)
        k2 = deriv(state + 0.5 * self.dt * k1, u)
        k3 = deriv(state + 0.5 * self.dt * k2, u)
        k4 = deriv(state + self.dt * k3, u)
        return state + (self.dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

    def get_expert_mpc_action(self, state):
        th, dth = state
        u = - (self.K[0] * th + self.K[1] * dth)
        return float(np.clip(u, -self.u_max, self.u_max))

    def project_mpc_guidance(self, state, u_generative):
        u_expert = self.get_expert_mpc_action(state)
        u_guided = 0.45 * u_generative + 0.55 * u_expert
        return float(np.clip(u_guided, -self.u_max, self.u_max))

# 2. Neural Models
class ScoreDiffusionNet(nn.Module):
    def __init__(self, state_dim=2, action_dim=1, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim + 1, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, action_dim)
        )
    def forward(self, u_noisy, state, t):
        return self.net(torch.cat([u_noisy, state, t], dim=-1))

class FlowMatchingNet(nn.Module):
    def __init__(self, state_dim=2, action_dim=1, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim + 1, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, action_dim)
        )
    def forward(self, u_t, state, t):
        return self.net(torch.cat([u_t, state, t], dim=-1))

def evaluate():
    env = DynamicBalancingEnv()
    
    # Load Weights
    ddpm_net = ScoreDiffusionNet().to(device)
    ddpm_net.load_state_dict(torch.load("/Volumes/KIOXIA/difusionMPC/checkpoints/pure_ddpm_checkpoint.pt", map_location=device))
    ddpm_net.eval()
    
    cfm_net = FlowMatchingNet().to(device)
    cfm_net.load_state_dict(torch.load("/Volumes/KIOXIA/difusionMPC/checkpoints/pure_cfm_checkpoint.pt", map_location=device))
    cfm_net.eval()
    
    guided_net = FlowMatchingNet().to(device)
    guided_net.load_state_dict(torch.load("/Volumes/KIOXIA/difusionMPC/checkpoints/model_guided_diffusion_checkpoint.pt", map_location=device))
    guided_net.eval()
    
    print("Loaded all model checkpoints successfully from /Volumes/KIOXIA/difusionMPC/checkpoints/")
    
    # Diffusion parameters
    num_ddpm_steps = 15
    betas = torch.linspace(1e-4, 0.04, num_ddpm_steps, device=device)
    alphas = 1.0 - betas
    alphas_bar = torch.cumprod(alphas, dim=0)

    def sample_ddpm(state_np):
        with torch.no_grad():
            s = torch.as_tensor(state_np, dtype=torch.float32, device=device).unsqueeze(0)
            u = torch.randn(1, 1, device=device)
            for i in reversed(range(num_ddpm_steps)):
                t = torch.tensor([[i / num_ddpm_steps]], device=device)
                pred_eps = ddpm_net(u, s, t)
                beta = betas[i]
                alpha = alphas[i]
                alpha_bar = alphas_bar[i]
                u = (u - beta / torch.sqrt(1.0 - alpha_bar) * pred_eps) / torch.sqrt(alpha)
                if i > 0:
                    u = u + torch.sqrt(beta) * torch.randn_like(u)
        return float(u.cpu().numpy()[0, 0])

    def sample_cfm(state_np, net):
        with torch.no_grad():
            s = torch.as_tensor(state_np, dtype=torch.float32, device=device).unsqueeze(0)
            u = torch.randn(1, 1, device=device)
            steps = 8
            dt_step = 1.0 / steps
            for i in range(steps):
                t = torch.tensor([[i * dt_step]], device=device)
                v = net(u, s, t)
                u = u + v * dt_step
        return float(u.cpu().numpy()[0, 0])

    # 100-Trial Stress Test under High Disturbance Noise (sigma = 0.6 N*m)
    # and Large Unseen Angle Shocks (|th| in [0.25, 0.45] rad)
    print("\n" + "=" * 70)
    print("STRESS-TESTING LOADED CHECKPOINTS UNDER SEVERE PERTURBATION NOISE (sigma = 0.6 N*m)...")
    print("=" * 70)
    N_TEST = 100
    
    results = {
        'Pure_DDPM_Checkpoint': {'success': 0, 'tumble_loss': 0},
        'Pure_CFM_Checkpoint': {'success': 0, 'tumble_loss': 0},
        'Model_Guided_Diffusion (Ours)': {'success': 0, 'tumble_loss': 0}
    }
    
    for trial in range(N_TEST):
        np.random.seed(9000 + trial)
        # Large initial perturbation shock (beyond nominal 0.25 rad)
        init_th = np.random.uniform(-0.45, 0.45)
        init_dth = np.random.uniform(-0.35, 0.35)
        x_init = np.array([init_th, init_dth])
        
        # 1. Pure DDPM Checkpoint
        s = x_init.copy()
        fell = False
        for _ in range(120):
            u = sample_ddpm(s) + np.random.normal(0, 0.6) # High noise
            s = env.step(s, u)
            if np.abs(s[0]) > 0.8:
                fell = True
                break
        if fell:
            results['Pure_DDPM_Checkpoint']['tumble_loss'] += 1
        elif np.abs(s[0]) <= 0.05:
            results['Pure_DDPM_Checkpoint']['success'] += 1
            
        # 2. Pure CFM Checkpoint
        s = x_init.copy()
        fell = False
        for _ in range(120):
            u = sample_cfm(s, cfm_net) + np.random.normal(0, 0.6)
            s = env.step(s, u)
            if np.abs(s[0]) > 0.8:
                fell = True
                break
        if fell:
            results['Pure_CFM_Checkpoint']['tumble_loss'] += 1
        elif np.abs(s[0]) <= 0.05:
            results['Pure_CFM_Checkpoint']['success'] += 1
            
        # 3. Model-Guided Diffusion Checkpoint (Ours)
        s = x_init.copy()
        fell = False
        for _ in range(120):
            u_gen = sample_cfm(s, guided_net)
            u_guided = env.project_mpc_guidance(s, u_gen) + np.random.normal(0, 0.6)
            s = env.step(s, u_guided)
            if np.abs(s[0]) > 0.8:
                fell = True
                break
        if fell:
            results['Model_Guided_Diffusion (Ours)']['tumble_loss'] += 1
        elif np.abs(s[0]) <= 0.05:
            results['Model_Guided_Diffusion (Ours)']['success'] += 1

    print("\n" + "=" * 70)
    print("FINAL 100-TRIAL CHECKPOINT BENCHMARK RESULTS (HIGH NOISE STRESS TEST):")
    print("=" * 70)
    for model_name, stat in results.items():
        s_rate = (stat['success'] / N_TEST) * 100.0
        f_rate = (stat['tumble_loss'] / N_TEST) * 100.0
        print(f"  • {model_name:32s} | Success Rate: {s_rate:5.1f}% | Tumble Rate: {f_rate:5.1f}%")
    print("=" * 70)

if __name__ == "__main__":
    evaluate()
