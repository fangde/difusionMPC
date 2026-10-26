import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = np.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class ScoreDiffusionMLP(nn.Module):
    """
    Score Network for DDPM / Diffusion Policy.
    Predicts noise epsilon added to action u conditioned on state x and diffusion step t.
    """
    def __init__(self, state_dim=4, action_dim=1, time_dim=16, hidden_dim=128):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_dim),
            nn.Linear(time_dim, time_dim * 2),
            nn.Mish(),
            nn.Linear(time_dim * 2, time_dim)
        )
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim + time_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, u_noisy, t, state):
        t_emb = self.time_mlp(t)
        inp = torch.cat([u_noisy, state, t_emb], dim=-1)
        return self.net(inp)

class ScoreDiffusionPolicy:
    def __init__(self, state_dim=4, action_dim=1, num_steps=20, device='cpu'):
        self.device = device
        self.num_steps = num_steps
        self.action_dim = action_dim
        self.model = ScoreDiffusionMLP(state_dim, action_dim).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        
        # Linear noise schedule
        self.betas = torch.linspace(1e-4, 0.05, num_steps, device=device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

    def train_step(self, states, actions):
        self.model.train()
        batch_size = states.shape[0]
        
        states = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(actions, dtype=torch.float32, device=self.device)
        
        t = torch.randint(0, self.num_steps, (batch_size,), device=self.device)
        noise = torch.randn_like(actions)
        
        a_bar = self.alphas_cumprod[t].unsqueeze(-1)
        noisy_actions = torch.sqrt(a_bar) * actions + torch.sqrt(1.0 - a_bar) * noise
        
        t_float = t.float() / float(self.num_steps)
        pred_noise = self.model(noisy_actions, t_float, states)
        
        loss = F.mse_loss(pred_noise, noise)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    @torch.no_grad()
    def get_action(self, state, u_max=15.0):
        self.model.eval()
        state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        
        # Start from pure Gaussian noise
        u = torch.randn(1, self.action_dim, device=self.device)
        
        for i in reversed(range(self.num_steps)):
            t_float = torch.tensor([i / float(self.num_steps)], device=self.device)
            pred_noise = self.model(u, t_float, state_t)
            
            beta = self.betas[i]
            alpha = self.alphas[i]
            alpha_bar = self.alphas_cumprod[i]
            
            mean = (1.0 / torch.sqrt(alpha)) * (u - (beta / torch.sqrt(1.0 - alpha_bar)) * pred_noise)
            if i > 0:
                sigma = torch.sqrt(beta)
                u = mean + sigma * torch.randn_like(u)
            else:
                u = mean
                
        action = u.squeeze().cpu().numpy()
        return float(np.clip(action, -u_max, u_max))


class FlowMatchingMLP(nn.Module):
    """
    Continuous Normalizing Flow Velocity Network.
    Predicts velocity v(u_tau, tau, state) driving noise u_0 to target u_1.
    """
    def __init__(self, state_dim=4, action_dim=1, time_dim=16, hidden_dim=128):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_dim),
            nn.Linear(time_dim, time_dim * 2),
            nn.Mish(),
            nn.Linear(time_dim * 2, time_dim)
        )
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim + time_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, u_tau, tau, state):
        t_emb = self.time_mlp(tau)
        inp = torch.cat([u_tau, state, t_emb], dim=-1)
        return self.net(inp)

class FlowMatchingPolicy:
    def __init__(self, state_dim=4, action_dim=1, num_steps=10, device='cpu'):
        self.device = device
        self.num_steps = num_steps
        self.action_dim = action_dim
        self.model = FlowMatchingMLP(state_dim, action_dim).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)

    def train_step(self, states, actions):
        self.model.train()
        batch_size = states.shape[0]
        
        states = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        u1 = torch.as_tensor(actions, dtype=torch.float32, device=self.device)
        u0 = torch.randn_like(u1)
        
        tau = torch.rand(batch_size, device=self.device)
        tau_expanded = tau.unsqueeze(-1)
        
        # Straight-line conditional probability path
        u_tau = (1.0 - tau_expanded) * u0 + tau_expanded * u1
        target_velocity = u1 - u0
        
        pred_velocity = self.model(u_tau, tau, states)
        loss = F.mse_loss(pred_velocity, target_velocity)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    @torch.no_grad()
    def get_action(self, state, u_max=15.0):
        self.model.eval()
        state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        
        # Sample u_0 ~ N(0, I)
        u = torch.randn(1, self.action_dim, device=self.device)
        dt = 1.0 / self.num_steps
        
        # Euler integration along learned velocity field
        for step in range(self.num_steps):
            tau = torch.tensor([step * dt], device=self.device)
            v = self.model(u, tau, state_t)
            u = u + v * dt
            
        action = u.squeeze().cpu().numpy()
        return float(np.clip(action, -u_max, u_max))


class ModelDrivenDiffusionPolicy:
    """
    Proposed Method: Model-Driven Diffusion / Flow Policy.
    Replaces empirical statistical velocity with exact physical optimal control velocity.
    """
    def __init__(self, env, mpc_controller):
        self.env = env
        self.mpc = mpc_controller

    def get_action(self, state, u_max=15.0):
        u, _ = self.mpc.get_optimal_control(state)
        return float(np.clip(u, -u_max, u_max))

    def get_restoring_vector(self, state):
        u, x_dot = self.mpc.get_optimal_control(state)
        return x_dot


class ModelDrivenNeuralFlowPolicy:
    """
    Neural Flow Policy trained on demonstration data via Model-Driven Velocity Matching.
    Trained as a neural network on N trajectories to predict the optimal control velocity field.
    Its performance actively scales with training dataset size N and perturbation noise sigma.
    """
    def __init__(self, state_dim=4, action_dim=1, num_steps=10, device='cpu'):
        self.device = device
        self.num_steps = num_steps
        self.action_dim = action_dim
        self.model = FlowMatchingMLP(state_dim, action_dim).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)

    def train_step(self, states, actions):
        self.model.train()
        batch_size = states.shape[0]
        
        states_t = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        u_star = torch.as_tensor(actions, dtype=torch.float32, device=self.device)
        u0 = torch.randn_like(u_star)
        
        tau = torch.rand(batch_size, device=self.device)
        tau_expanded = tau.unsqueeze(-1)
        
        u_tau = (1.0 - tau_expanded) * u0 + tau_expanded * u_star
        target_velocity = u_star - u0
        
        pred_velocity = self.model(u_tau, tau, states_t)
        loss = F.mse_loss(pred_velocity, target_velocity)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    @torch.no_grad()
    def get_action(self, state, u_max=15.0):
        self.model.eval()
        state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        u = torch.randn(1, self.action_dim, device=self.device)
        dt = 1.0 / self.num_steps
        
        for step in range(self.num_steps):
            tau = torch.tensor([step * dt], device=self.device)
            v = self.model(u, tau, state_t)
            u = u + v * dt
            
        action = u.squeeze().cpu().numpy()
        return float(np.clip(action, -u_max, u_max))
