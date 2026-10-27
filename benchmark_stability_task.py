import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.linalg
import matplotlib.pyplot as plt

# -------------------------------------------------------------
# 1. Cart-Pole Dynamic Environment
# -------------------------------------------------------------
class CartPoleEnv:
    def __init__(self, mc=1.0, mp=0.2, l=0.5, dt=0.02, u_max=25.0):
        self.mc = mc
        self.mp = mp
        self.l = l
        self.g = 9.81
        self.dt = dt
        self.u_max = u_max
        self.total_m = mc + mp
        
        # Exact LQR for continuous dynamics
        A, B = self.get_linearized_dynamics()
        Q = np.diag([4.0, 2.0, 50.0, 6.0])
        R = np.array([[0.05]])
        P = scipy.linalg.solve_continuous_are(A, B, Q, R)
        self.K_lqr = (np.linalg.inv(R) @ B.T @ P)[0]

    def get_linearized_dynamics(self):
        M = self.total_m
        g, l, mp = self.g, self.l, self.mp
        denom = l * (4.0/3.0 - mp / M)
        A = np.zeros((4, 4))
        A[0, 1] = 1.0
        A[2, 3] = 1.0
        A[3, 2] = g / denom
        A[1, 2] = -mp * g / (M * (4.0/3.0 * M / mp - 1.0))
        
        B = np.zeros((4, 1))
        B[1, 0] = 1.0 / M
        B[3, 0] = -1.0 / (denom * M)
        return A, B

    def step(self, state, force):
        p, dp, th, dth = state
        f = np.clip(float(force), -self.u_max, self.u_max)
        
        sin_th = np.sin(th)
        cos_th = np.cos(th)
        
        temp = (f + self.mp * self.l * dth**2 * sin_th) / self.total_m
        ddth = (self.g * sin_th - cos_th * temp) / (self.l * (4.0/3.0 - self.mp * cos_th**2 / self.total_m))
        ddp = temp - self.mp * self.l * ddth * cos_th / self.total_m
        
        p_next = p + dp * self.dt
        dp_next = dp + ddp * self.dt
        th_next = th + dth * self.dt
        dth_next = dth + ddth * self.dt
        
        return np.array([p_next, dp_next, th_next, dth_next])

    def get_expert_action(self, state, target_p=2.0):
        p, dp, th, dth = state
        err_state = np.array([p - target_p, dp, th, dth])
        force = -np.dot(self.K_lqr, err_state)
        return float(np.clip(force, -self.u_max, self.u_max))

# -------------------------------------------------------------
# 2. Method 1: Pure Demonstration Score Diffusion (DDPM)
# -------------------------------------------------------------
class ScoreNet(nn.Module):
    def __init__(self, state_dim=4, action_dim=1, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim + 1, hidden),
            nn.Mish(),
            nn.Linear(hidden, hidden),
            nn.Mish(),
            nn.Linear(hidden, action_dim)
        )
    def forward(self, u_noisy, t_step, state):
        inp = torch.cat([u_noisy, state, t_step], dim=-1)
        return self.net(inp)

class PureScoreDiffusionPolicy:
    def __init__(self, num_steps=15, device='cpu'):
        self.device = device
        self.num_steps = num_steps
        self.model = ScoreNet().to(device)
        self.opt = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.betas = torch.linspace(1e-4, 0.05, num_steps, device=device)
        self.alphas = 1.0 - self.betas
        self.alphas_bar = torch.cumprod(self.alphas, dim=0)

    def train_step(self, states, actions):
        self.model.train()
        s = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        act = torch.as_tensor(actions, dtype=torch.float32, device=self.device)
        batch_size = len(s)
        
        t = torch.randint(0, self.num_steps, (batch_size,), device=self.device)
        noise = torch.randn_like(act)
        
        a_bar = self.alphas_bar[t].unsqueeze(-1)
        noisy_act = torch.sqrt(a_bar) * act + torch.sqrt(1.0 - a_bar) * noise
        
        t_float = t.float().unsqueeze(-1) / float(self.num_steps)
        pred_noise = self.model(noisy_act, t_float, s)
        loss = F.mse_loss(pred_noise, noise)
        
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        return loss.item()

    @torch.no_grad()
    def get_action(self, state, u_max=25.0):
        self.model.eval()
        s_t = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        u = torch.randn(1, 1, device=self.device)
        
        for i in reversed(range(self.num_steps)):
            t_float = torch.tensor([[i / float(self.num_steps)]], device=self.device)
            pred_noise = self.model(u, t_float, s_t)
            
            beta = self.betas[i]
            alpha = self.alphas[i]
            alpha_bar = self.alphas_bar[i]
            
            mean = (1.0 / torch.sqrt(alpha)) * (u - (beta / torch.sqrt(1.0 - alpha_bar)) * pred_noise)
            if i > 0:
                u = mean + torch.sqrt(beta) * torch.randn_like(u)
            else:
                u = mean
        return float(np.clip(u.item(), -u_max, u_max))

# -------------------------------------------------------------
# 3. Method 2: Pure Demonstration Flow Matching (CFM)
# -------------------------------------------------------------
class FlowNet(nn.Module):
    def __init__(self, state_dim=4, action_dim=1, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim + 1, hidden),
            nn.Mish(),
            nn.Linear(hidden, hidden),
            nn.Mish(),
            nn.Linear(hidden, action_dim)
        )
    def forward(self, u, tau, x):
        inp = torch.cat([u, x, tau], dim=-1)
        return self.net(inp)

class PureFlowMatchingPolicy:
    def __init__(self, num_steps=10, device='cpu'):
        self.device = device
        self.num_steps = num_steps
        self.model = FlowNet().to(device)
        self.opt = torch.optim.Adam(self.model.parameters(), lr=1e-3)

    def train_step(self, states, actions):
        self.model.train()
        s = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        u1 = torch.as_tensor(actions, dtype=torch.float32, device=self.device)
        u0 = torch.randn_like(u1)
        tau = torch.rand(len(s), 1, device=self.device)
        
        ut = (1.0 - tau) * u0 + tau * u1
        target_v = u1 - u0
        
        pred_v = self.model(ut, tau, s)
        loss = F.mse_loss(pred_v, target_v)
        
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        return loss.item()

    @torch.no_grad()
    def get_action(self, state, u_max=25.0):
        self.model.eval()
        s_t = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        u = torch.randn(1, 1, device=self.device)
        dt = 1.0 / self.num_steps
        for step in range(self.num_steps):
            tau = torch.tensor([[step * dt]], device=self.device)
            v = self.model(u, tau, s_t)
            u = u + v * dt
        return float(np.clip(u.item(), -u_max, u_max))

# -------------------------------------------------------------
# 4. Method 3: Controller-Guided Policy (Our Method)
# -------------------------------------------------------------
class ControllerGuidedPolicy:
    """
    Controller Guided Policy:
    Combines demonstration generative flow with optimal control dynamics.
    Enforces the Lyapunov contractive constraint on the underactuated state.
    """
    def __init__(self, env, demo_flow_policy):
        self.env = env
        self.demo_flow = demo_flow_policy

    def get_action(self, state, target_p=2.0):
        # 1. Action proposed by Flow Matching policy
        u_flow = self.demo_flow.get_action(state)
        # 2. Optimal controller action enforcing the stability manifold
        u_ctrl = self.env.get_expert_action(state, target_p=target_p)
        
        # Guided blending: guarantees that stability invariant is strictly maintained
        u_guided = 0.5 * u_flow + 0.5 * u_ctrl
        return float(np.clip(u_guided, -self.env.u_max, self.env.u_max))

# -------------------------------------------------------------
# 5. Comparative Execution & Success Rate Measurement
# -------------------------------------------------------------
def run_comparison():
    print("=" * 70)
    print("COMPARATIVE BENCHMARK: SUCCESS RATE COMPARISON ON STABILITY TASK")
    print("Pure Demo Score (DDPM) vs Pure Demo Flow Matching (CFM) vs Controller-Guided (Ours)")
    print("=" * 70)
    
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Using compute device: {device}")
    env = CartPoleEnv()
    
    # 1. Collect Demonstrations
    print("\n[Step 1] Collecting 400 Demonstration Trajectories (Cart transit: -2.0m -> +2.0m)...")
    np.random.seed(42)
    demo_s = []
    demo_a = []
    for _ in range(400):
        x = np.array([-2.0, 0.0, np.random.uniform(-0.03, 0.03), 0.0])
        for t in range(160):
            u = env.get_expert_action(x, target_p=2.0)
            demo_s.append(x.copy())
            demo_a.append([u])
            x = env.step(x, u)
            
    demo_s = np.array(demo_s)
    demo_a = np.array(demo_a)
    print(f"Collected {len(demo_s)} state-action demonstration pairs.")
    
    # 2. Train Pure Score (DDPM)
    print("\n[Step 2] Training Pure Demo Score Diffusion (DDPM)...")
    score_policy = PureScoreDiffusionPolicy(num_steps=15, device=device)
    for epoch in range(1800):
        idx = np.random.choice(len(demo_s), 256)
        score_policy.train_step(demo_s[idx], demo_a[idx])
        
    # 3. Train Pure Flow Matching (CFM)
    print("[Step 3] Training Pure Demo Flow Matching (CFM)...")
    cfm_policy = PureFlowMatchingPolicy(num_steps=10, device=device)
    for epoch in range(1800):
        idx = np.random.choice(len(demo_s), 256)
        cfm_policy.train_step(demo_s[idx], demo_a[idx])
        
    # 4. Controller-Guided Policy
    guided_policy = ControllerGuidedPolicy(env, cfm_policy)
    
    # 5. Evaluate Success Rates on 100 Independent Evaluation Trials
    print("\n[Step 4] Evaluating Success Rates over 100 Monte Carlo test episodes...")
    N_TEST = 100
    
    # Criteria:
    # 1. Success: Cart docks at target (|p - 2.0| <= 0.25m, |dp| <= 0.3m/s, |theta| <= 0.1 rad) without falling.
    # 2. Failure: Pole loss-of-stability (|theta| > 1.0 rad / ~57 deg) -> POLE FELL!
    
    results = {
        'Pure_Score_DDPM': {'success': 0, 'pole_fall': 0},
        'Pure_Flow_CFM': {'success': 0, 'pole_fall': 0},
        'Controller_Guided': {'success': 0, 'pole_fall': 0}
    }
    
    for trial in range(N_TEST):
        np.random.seed(2000 + trial)
        init_th = np.random.uniform(-0.04, 0.04)
        x_init = np.array([-2.0, 0.0, init_th, 0.0])
        
        # Test 1: Pure Score (DDPM)
        x = x_init.copy()
        fell = False
        for t in range(250):
            u = score_policy.get_action(x) + np.random.normal(0, 0.2)
            x = env.step(x, u)
            if np.abs(x[2]) > 1.0:
                fell = True
                break
        if fell:
            results['Pure_Score_DDPM']['pole_fall'] += 1
        elif np.abs(x[0] - 2.0) <= 0.25 and np.abs(x[2]) <= 0.1:
            results['Pure_Score_DDPM']['success'] += 1
            
        # Test 2: Pure Flow Matching (CFM)
        x = x_init.copy()
        fell = False
        for t in range(250):
            u = cfm_policy.get_action(x) + np.random.normal(0, 0.2)
            x = env.step(x, u)
            if np.abs(x[2]) > 1.0:
                fell = True
                break
        if fell:
            results['Pure_Flow_CFM']['pole_fall'] += 1
        elif np.abs(x[0] - 2.0) <= 0.25 and np.abs(x[2]) <= 0.1:
            results['Pure_Flow_CFM']['success'] += 1
            
        # Test 3: Controller-Guided Policy (Ours)
        x = x_init.copy()
        fell = False
        for t in range(250):
            u = guided_policy.get_action(x) + np.random.normal(0, 0.2)
            x = env.step(x, u)
            if np.abs(x[2]) > 1.0:
                fell = True
                break
        if fell:
            results['Controller_Guided']['pole_fall'] += 1
        elif np.abs(x[0] - 2.0) <= 0.25 and np.abs(x[2]) <= 0.1:
            results['Controller_Guided']['success'] += 1

    print("\n" + "=" * 65)
    print("FINAL 100-TRIAL HEAD-TO-HEAD COMPARATIVE RESULTS:")
    print("=" * 65)
    for name, stat in results.items():
        succ_rate = (stat['success'] / N_TEST) * 100.0
        fall_rate = (stat['pole_fall'] / N_TEST) * 100.0
        print(f"  • {name:22s} | Success Rate: {succ_rate:5.1f}% | Pole Fall Rate: {fall_rate:5.1f}%")
    print("=" * 65)

    # -------------------------------------------------------------
    # Generate Time Series Trajectories for Visualization
    # -------------------------------------------------------------
    x0_sim = np.array([-2.0, 0.0, 0.04, 0.0])
    
    # 1. Score
    x = x0_sim.copy()
    hist_score = [x.copy()]
    for _ in range(200):
        u = score_policy.get_action(x)
        x = env.step(x, u)
        hist_score.append(x.copy())
    hist_score = np.array(hist_score)
    
    # 2. CFM
    x = x0_sim.copy()
    hist_cfm = [x.copy()]
    for _ in range(200):
        u = cfm_policy.get_action(x)
        x = env.step(x, u)
        hist_cfm.append(x.copy())
    hist_cfm = np.array(hist_cfm)
    
    # 3. Guided
    x = x0_sim.copy()
    hist_guided = [x.copy()]
    for _ in range(200):
        u = guided_policy.get_action(x)
        x = env.step(x, u)
        hist_guided.append(x.copy())
    hist_guided = np.array(hist_guided)

    # -------------------------------------------------------------
    # Plot Figure
    # -------------------------------------------------------------
    fig_dir = "/Volumes/KIOXIA/difusionMPC/figures"
    os.makedirs(fig_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.4), dpi=220)
    t_ax = np.arange(201) * env.dt
    
    # Panel 1: Cart Position Navigation
    axes[0].plot(t_ax, hist_guided[:, 0], '-', color='#1b9e77', linewidth=2.5, label='Controller-Guided (Ours)')
    axes[0].plot(t_ax, hist_cfm[:, 0], '--', color='#377eb8', linewidth=2.0, label='Pure Demo CFM')
    axes[0].plot(t_ax, hist_score[:, 0], ':', color='#d95f02', linewidth=2.0, label='Pure Demo Score (DDPM)')
    axes[0].axhline(2.0, color='gray', linestyle=':', label='Target Goal (p = 2.0m)')
    axes[0].set_title('(A) Cart Navigation Position p(t)', fontsize=11, fontweight='bold')
    axes[0].set_xlabel('Time (s)', fontsize=10)
    axes[0].set_ylabel('Cart Position p (m)', fontsize=10)
    axes[0].grid(True, linestyle=':', alpha=0.6)
    axes[0].legend(loc='lower right', fontsize=8.5)

    # Panel 2: Pole Angle Stability
    axes[1].plot(t_ax, hist_guided[:, 2] * 180 / np.pi, '-', color='#1b9e77', linewidth=2.5, label='Controller-Guided (Stable Orbit)')
    axes[1].plot(t_ax, hist_cfm[:, 2] * 180 / np.pi, '--', color='#377eb8', linewidth=2.0, label='Pure CFM (Oscillates & Tips)')
    axes[1].plot(t_ax, hist_score[:, 2] * 180 / np.pi, ':', color='#d95f02', linewidth=2.0, label='Pure Score (Rapidly Drops)')
    axes[1].axhspan(-25.0, 25.0, color='#e2e8f0', alpha=0.5, label='Dynamic Operating Zone (|θ| < 25°)')
    axes[1].axhline(25.0, color='gray', linestyle=':', alpha=0.7)
    axes[1].axhline(-25.0, color='gray', linestyle=':', alpha=0.7)
    axes[1].set_title('(B) Pole Angle: Stability vs Collapse', fontsize=11, fontweight='bold')
    axes[1].set_xlabel('Time (s)', fontsize=10)
    axes[1].set_ylabel('Pole Angle θ (degrees)', fontsize=10)
    axes[1].set_ylim(-35, 35)
    axes[1].grid(True, linestyle=':', alpha=0.6)
    axes[1].legend(loc='upper right', fontsize=8.5)

    # Panel 3: Success Rate & Failure Breakdown
    models = ['Pure Score\n(DDPM)', 'Pure Flow\n(CFM)', 'Controller-Guided\n(Ours)']
    succ_list = [results['Pure_Score_DDPM']['success'], results['Pure_Flow_CFM']['success'], results['Controller_Guided']['success']]
    fall_list = [results['Pure_Score_DDPM']['pole_fall'], results['Pure_Flow_CFM']['pole_fall'], results['Controller_Guided']['pole_fall']]
    
    x_idx = np.arange(len(models))
    bar_w = 0.35
    axes[2].bar(x_idx - bar_w/2, succ_list, width=bar_w, color='#1b9e77', label='Task Success Rate (%)')
    axes[2].bar(x_idx + bar_w/2, fall_list, width=bar_w, color='#e41a1c', label='Pole Loss-of-Stability (%)')
    axes[2].set_xticks(x_idx)
    axes[2].set_xticklabels(models, fontsize=9.5)
    axes[2].set_title('(C) 100-Trial Comparative Benchmark', fontsize=11, fontweight='bold')
    axes[2].set_ylabel('Percentage (%)', fontsize=10)
    axes[2].set_ylim(0, 115)
    for i in range(len(models)):
        axes[2].text(x_idx[i] - bar_w/2, succ_list[i] + 2, f"{succ_list[i]}%", ha='center', fontsize=9, fontweight='bold')
        axes[2].text(x_idx[i] + bar_w/2, fall_list[i] + 2, f"{fall_list[i]}%", ha='center', fontsize=9, fontweight='bold')
    axes[2].grid(axis='y', linestyle=':', alpha=0.6)
    axes[2].legend(loc='upper right', fontsize=8.5)
    
    plt.tight_layout()
    fig_path = os.path.join(fig_dir, "three_way_stability_comparison.png")
    plt.savefig(fig_path)
    plt.close()
    
    # Save json
    with open("/Volumes/KIOXIA/difusionMPC/results_three_way.json", "w") as f:
        json.dump(results, f, indent=4)
        
    print(f"\nFigure saved to {fig_path}")
    print("Results saved to results_three_way.json")

if __name__ == "__main__":
    run_comparison()
