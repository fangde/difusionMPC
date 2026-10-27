import json
import os
import time
import numpy as np
import scipy.linalg
from scipy.optimize import lsq_linear
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ==============================================================================
# RIGOROUS 4-WAY BENCHMARK: MPC vs. BC vs. CFM vs. FLOW MATCHING + LYAPUNOV/CBF
#
# Addresses the peer critique:
# 1. Identical dynamics, identical initial conditions, identical disturbance noise.
# 2. Real neural networks (PyTorch) trained on identical dataset.
# 3. Mathematically rigorous invariant control set formulation:
#    S = {x : V(x) <= c}
#    U_stable(x) = {u : \dot{V}(x, u) <= -\alpha V(x)}
#    u* = argmin_{u \in U_stable} ||u - u_gen||^2
# 4. Measured metrics:
#    - Recovery Success Rate
#    - Tumble / Fall Rate
#    - Constraint Violation Rate
#    - Control Energy
#    - Latency per step (ms)
# ==============================================================================

device = torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))
# Use CPU for small batch inference to avoid GPU kernel launch overhead during per-step control
eval_device = torch.device('cpu')
print(f"[Device] Training on: {device}, Inference on: {eval_device}")

# ------------------------------------------------------------------------------
# 1. Physical Environment & Dynamics
# ------------------------------------------------------------------------------
class DynamicPendulumEnv:
    def __init__(self, m=1.0, l=1.0, g=9.81, dt=0.02, u_max=15.0):
        self.m = m
        self.l = l
        self.g = g
        self.dt = dt
        self.u_max = u_max
        self.I = m * l**2

        # Linearized Continuous Dynamics around upright [0, 0]^T:
        # \dot{x} = A_c x + B_c u
        self.A_c = np.array([[0.0, 1.0], [g / l, 0.0]])
        self.B_c = np.array([[0.0], [1.0 / self.I]])

        # Continuous CARE for Lyapunov function V(x) = x^T P x
        self.Q_care = np.diag([20.0, 2.0])
        self.R_care = np.array([[0.5]])
        self.P_care = scipy.linalg.solve_continuous_are(self.A_c, self.B_c, self.Q_care, self.R_care)
        self.K_lqr = np.linalg.inv(self.R_care) @ self.B_c.T @ self.P_care
        self.alpha_lyap = 1.2  # Contractive rate: \dot{V} <= -\alpha V

        # Discretized Dynamics for MPC Horizon
        self.A_d = scipy.linalg.expm(self.A_c * dt)
        self.B_d = np.linalg.inv(self.A_c) @ (self.A_d - np.eye(2)) @ self.B_c
        self.P_dare = scipy.linalg.solve_discrete_are(self.A_d, self.B_d, self.Q_care, self.R_care)

    def nonlinear_deriv(self, state, u):
        th, dth = state
        ddth = (self.g / self.l) * np.sin(th) + (u / self.I)
        return np.array([dth, ddth])

    def step(self, state, u, disturbance_noise=0.0):
        u_eff = float(np.clip(u + disturbance_noise, -self.u_max, self.u_max))
        # RK4 Integration
        k1 = self.nonlinear_deriv(state, u_eff)
        k2 = self.nonlinear_deriv(state + 0.5 * self.dt * k1, u_eff)
        k3 = self.nonlinear_deriv(state + 0.5 * self.dt * k2, u_eff)
        k4 = self.nonlinear_deriv(state + self.dt * k3, u_eff)
        next_state = state + (self.dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        return next_state

    def lyapunov_V(self, state):
        return float(state.T @ self.P_care @ state)

    def lyapunov_V_dot(self, state, u):
        # \dot{V}(x, u) = \nabla V(x)^T [f(x) + g(x) u]
        grad_V = 2.0 * self.P_care @ state
        f_x = np.array([state[1], (self.g / self.l) * np.sin(state[0])])
        g_x = np.array([0.0, 1.0 / self.I])
        return float(grad_V @ (f_x + g_x * u))

    def project_lyapunov_cbf(self, state, u_gen):
        """
        Exact minimum-intervention projection onto contractive set U_stable(x):
        u* = argmin_{|u| <= u_max} ||u - u_gen||^2
        s.t. \dot{V}(x, u) <= -\alpha V(x)
        """
        grad_V = 2.0 * self.P_care @ state
        f_x = np.array([state[1], (self.g / self.l) * np.sin(state[0])])
        g_x = np.array([0.0, 1.0 / self.I])
        
        # a(x) * u <= b(x)
        a = float(grad_V @ g_x)
        V_val = float(state.T @ self.P_care @ state)
        b = float(-self.alpha_lyap * V_val - grad_V @ f_x)

        if abs(a) < 1e-6:
            # Actuation does not affect derivative along this manifold direction
            return float(np.clip(u_gen, -self.u_max, self.u_max)), False

        # Invariant half-space bounds:
        # If a > 0: u <= b / a
        # If a < 0: u >= b / a
        bound = b / a
        if a > 0:
            u_upper = min(self.u_max, bound)
            u_lower = -self.u_max
        else:
            u_upper = self.u_max
            u_lower = max(-self.u_max, bound)

        if u_lower > u_upper:
            # Control constraint infeasibility (underactuated saturation limit)
            u_proj = float(np.clip(bound, -self.u_max, self.u_max))
            return u_proj, True # Infeasible flag

        u_proj = float(np.clip(u_gen, u_lower, u_upper))
        return u_proj, False

    def solve_mpc(self, state, horizon=12):
        """
        Receding Horizon Box-Constrained QP:
        min sum_{k=0}^{H-1} (x_k^T Q x_k + u_k^T R u_k) + x_H^T P x_H
        s.t. x_{k+1} = A_d x_k + B_d u_k, -u_max <= u_k <= u_max
        """
        H = horizon
        nx = 2
        nu = 1
        
        # Formulate condensed QP: X = Phi * x0 + Gamma * U
        Phi = np.zeros((H * nx, nx))
        Gamma = np.zeros((H * nx, H * nu))
        
        A_power = np.eye(nx)
        for i in range(H):
            A_power = A_power @ self.A_d
            Phi[i*nx:(i+1)*nx, :] = A_power
            for j in range(i + 1):
                A_step = np.linalg.matrix_power(self.A_d, i - j)
                Gamma[i*nx:(i+1)*nx, j*nu:(j+1)*nu] = A_step @ self.B_d
                
        # Cost matrices
        Q_bar = np.kron(np.eye(H), self.Q_care)
        Q_bar[(H-1)*nx:, (H-1)*nx:] = self.P_dare
        R_bar = np.kron(np.eye(H), self.R_care)
        
        # Condensed Hessian and Gradient:
        # 0.5 * U^T (2 * (R_bar + Gamma^T Q_bar Gamma)) U + (2 * Gamma^T Q_bar Phi x0)^T U
        Hess = 2.0 * (R_bar + Gamma.T @ Q_bar @ Gamma)
        grad = 2.0 * (Gamma.T @ Q_bar @ Phi @ state)
        
        # Solve via Cholesky + Box-constrained Least Squares
        try:
            L = np.linalg.cholesky(Hess)
            d = -np.linalg.solve(L, grad)
            res = lsq_linear(L.T, d, bounds=(-self.u_max, self.u_max), lsmr_tol='auto', verbose=0)
            u_opt = float(res.x[0])
        except Exception:
            # Fallback to unconstrained LQR if numerical QP failure
            u_opt = float(-self.K_lqr @ state)
        return float(np.clip(u_opt, -self.u_max, self.u_max))

# ------------------------------------------------------------------------------
# 2. Real PyTorch Neural Architectures
# ------------------------------------------------------------------------------
class DeterministicBCNet(nn.Module):
    """Supervised Behavioral Cloning MLP"""
    def __init__(self, state_dim=2, action_dim=1, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, action_dim)
        )
    def forward(self, state):
        return self.net(state)

class FlowMatchingNet(nn.Module):
    """Conditional Flow Matching (CFM) Vector Field Network"""
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

# ------------------------------------------------------------------------------
# 3. Data Generation & Model Training
# ------------------------------------------------------------------------------
def generate_dataset_and_train():
    os.makedirs("/Volumes/KIOXIA/difusionMPC/data", exist_ok=True)
    os.makedirs("/Volumes/KIOXIA/difusionMPC/checkpoints", exist_ok=True)
    env = DynamicPendulumEnv()
    
    print("\n[Step 1/3] Generating 40,000 physical transition samples...")
    np.random.seed(42)
    N_SAMPLES = 40000
    states = []
    actions = []
    
    for _ in range(N_SAMPLES):
        # Sample broad state space including nominal and exploration boundary
        th = np.random.uniform(-0.55, 0.55)
        dth = np.random.uniform(-1.2, 1.2)
        s = np.array([th, dth])
        
        # MPC expert target action with small exploration noise
        u_mpc = env.solve_mpc(s, horizon=10)
        u_noisy = u_mpc + np.random.normal(0, 0.2)
        u_final = float(np.clip(u_noisy, -env.u_max, env.u_max))
        
        states.append(s)
        actions.append(u_final)
        
    states_np = np.array(states, dtype=np.float32)
    actions_np = np.array(actions, dtype=np.float32).reshape(-1, 1)
    
    np.save("/Volumes/KIOXIA/difusionMPC/data/four_way_train_states.npy", states_np)
    np.save("/Volumes/KIOXIA/difusionMPC/data/four_way_train_actions.npy", actions_np)
    print(f"  -> Saved dataset: states {states_np.shape}, actions {actions_np.shape}")
    
    # Dataset tensors
    states_tensor = torch.tensor(states_np, dtype=torch.float32).to(device)
    actions_tensor = torch.tensor(actions_np, dtype=torch.float32).to(device)
    dataset = torch.utils.data.TensorDataset(states_tensor, actions_tensor)
    loader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=True)
    
    print("\n[Step 2/3] Training Deterministic BC Model (MLP)...")
    bc_model = DeterministicBCNet().to(device)
    optimizer_bc = torch.optim.AdamW(bc_model.parameters(), lr=1e-3, weight_decay=1e-4)
    
    bc_model.train()
    for epoch in range(15):
        total_loss = 0.0
        for s_b, a_b in loader:
            optimizer_bc.zero_grad()
            pred_a = bc_model(s_b)
            loss = F.mse_loss(pred_a, a_b)
            loss.backward()
            optimizer_bc.step()
            total_loss += loss.item()
        if (epoch + 1) % 5 == 0:
            print(f"  [BC] Epoch {epoch+1:2d}/15 | MSE Loss: {total_loss / len(loader):.5f}")
            
    torch.save(bc_model.state_dict(), "/Volumes/KIOXIA/difusionMPC/checkpoints/bc_model.pt")
    print("  -> Saved BC model to /Volumes/KIOXIA/difusionMPC/checkpoints/bc_model.pt")
    
    print("\n[Step 3/3] Training Flow Matching Model (CFM)...")
    cfm_model = FlowMatchingNet().to(device)
    optimizer_cfm = torch.optim.AdamW(cfm_model.parameters(), lr=1e-3, weight_decay=1e-4)
    
    cfm_model.train()
    for epoch in range(20):
        total_loss = 0.0
        for s_b, a_1 in loader:
            optimizer_cfm.zero_grad()
            B = s_b.shape[0]
            t = torch.rand(B, 1, device=device)
            a_0 = torch.randn_like(a_1) # Gaussian prior
            
            # Flow matching interpolation
            a_t = (1.0 - t) * a_0 + t * a_1
            target_v = a_1 - a_0
            
            pred_v = cfm_model(a_t, s_b, t)
            loss = F.mse_loss(pred_v, target_v)
            loss.backward()
            optimizer_cfm.step()
            total_loss += loss.item()
        if (epoch + 1) % 5 == 0:
            print(f"  [CFM] Epoch {epoch+1:2d}/20 | Velocity MSE Loss: {total_loss / len(loader):.5f}")
            
    torch.save(cfm_model.state_dict(), "/Volumes/KIOXIA/difusionMPC/checkpoints/cfm_model.pt")
    # For model-guided diffusion, it shares the exact same trained CFM checkpoint as base generator!
    torch.save(cfm_model.state_dict(), "/Volumes/KIOXIA/difusionMPC/checkpoints/mpc_guided_cbf_model.pt")
    print("  -> Saved CFM checkpoints successfully.")

# ------------------------------------------------------------------------------
# 4. Rigorous 4-Way Monte Carlo Evaluation
# ------------------------------------------------------------------------------
def run_four_way_benchmark():
    env = DynamicPendulumEnv()
    
    # Load Models on CPU for single-instance per-step control
    bc_model = DeterministicBCNet().to(eval_device)
    bc_model.load_state_dict(torch.load("/Volumes/KIOXIA/difusionMPC/checkpoints/bc_model.pt", map_location=eval_device))
    bc_model.eval()
    
    cfm_model = FlowMatchingNet().to(eval_device)
    cfm_model.load_state_dict(torch.load("/Volumes/KIOXIA/difusionMPC/checkpoints/cfm_model.pt", map_location=eval_device))
    cfm_model.eval()
    
    def sample_cfm(state_np, num_steps=5):
        with torch.no_grad():
            s = torch.as_tensor(state_np, dtype=torch.float32, device=eval_device).unsqueeze(0)
            u = torch.randn(1, 1, device=eval_device)
            dt_step = 1.0 / num_steps
            for i in range(num_steps):
                t = torch.tensor([[i * dt_step]], dtype=torch.float32, device=eval_device)
                v = cfm_model(u, s, t)
                u = u + v * dt_step
        return float(u.numpy()[0, 0])

    def sample_bc(state_np):
        with torch.no_grad():
            s = torch.as_tensor(state_np, dtype=torch.float32, device=eval_device).unsqueeze(0)
            return float(bc_model(s).numpy()[0, 0])

    N_TEST_EPISODES = 100
    STEPS_PER_EPISODE = 150 # 3.0 seconds at dt=0.02s
    NOISE_SIGMA = 0.5       # Heavy process perturbation noise at every step (N*m)
    
    # Seed fixed initial disturbance conditions
    np.random.seed(2026)
    initial_states = []
    for _ in range(N_TEST_EPISODES):
        # Shock initial angles up to 0.40 rad (~23 deg) and angular velocity 0.45 rad/s
        th0 = np.random.uniform(-0.40, 0.40)
        dth0 = np.random.uniform(-0.45, 0.45)
        initial_states.append(np.array([th0, dth0]))
        
    noise_sequences = np.random.normal(0, NOISE_SIGMA, size=(N_TEST_EPISODES, STEPS_PER_EPISODE))

    controllers = [
        "Pure MPC (Receding Horizon Box-QP)",
        "Deterministic BC (MLP)",
        "Pure Flow Matching (CFM)",
        "Flow Matching + Lyapunov/CBF (Ours)"
    ]

    benchmark_stats = {
        c: {
            "success_count": 0,
            "tumble_count": 0,
            "constraint_violations": 0,
            "total_steps": 0,
            "total_energy": 0.0,
            "latencies_ms": []
        } for c in controllers
    }

    print("\n" + "=" * 78)
    print(f"RUNNING 100-TRIAL BENCHMARK (N={N_TEST_EPISODES}, Noise sigma={NOISE_SIGMA} N*m, Horizon={STEPS_PER_EPISODE} steps)")
    print("=" * 78)

    for ep in range(N_TEST_EPISODES):
        x0 = initial_states[ep].copy()
        ep_noises = noise_sequences[ep]
        
        for c_idx, c_name in enumerate(controllers):
            s = x0.copy()
            fell = False
            ep_energy = 0.0
            
            for t_idx in range(STEPS_PER_EPISODE):
                dist = ep_noises[t_idx]
                
                t_start = time.perf_counter()
                if c_name == "Pure MPC (Receding Horizon Box-QP)":
                    u = env.solve_mpc(s, horizon=10)
                    infeasible = False
                elif c_name == "Deterministic BC (MLP)":
                    u = float(np.clip(sample_bc(s), -env.u_max, env.u_max))
                    infeasible = False
                elif c_name == "Pure Flow Matching (CFM)":
                    u = float(np.clip(sample_cfm(s, num_steps=5), -env.u_max, env.u_max))
                    infeasible = False
                elif c_name == "Flow Matching + Lyapunov/CBF (Ours)":
                    u_raw = sample_cfm(s, num_steps=5)
                    u, infeasible = env.project_lyapunov_cbf(s, u_raw)
                t_elapsed = (time.perf_counter() - t_start) * 1000.0 # ms
                
                benchmark_stats[c_name]["latencies_ms"].append(t_elapsed)
                benchmark_stats[c_name]["total_steps"] += 1
                
                # Check Lyapunov Invariant condition violation: \dot{V} > -\alpha V
                v_dot = env.lyapunov_V_dot(s, u)
                V_val = env.lyapunov_V(s)
                if v_dot > -env.alpha_lyap * V_val + 1e-4:
                    benchmark_stats[c_name]["constraint_violations"] += 1
                
                # Accumulate energy: u^2 * dt
                ep_energy += (u ** 2) * env.dt
                
                # Step physics
                s = env.step(s, u, disturbance_noise=dist)
                
                # Check fall condition (|theta| > 0.80 rad ~ 46 degrees)
                if abs(s[0]) > 0.80:
                    fell = True
                    break
                    
            benchmark_stats[c_name]["total_energy"] += ep_energy
            
            if fell:
                benchmark_stats[c_name]["tumble_count"] += 1
            else:
                # Target Attractor: |theta| <= 0.06 rad and |omega| <= 0.15 rad/s
                if abs(s[0]) <= 0.06 and abs(s[1]) <= 0.15:
                    benchmark_stats[c_name]["success_count"] += 1
                    
        if (ep + 1) % 25 == 0:
            print(f"  -> Completed {ep + 1}/{N_TEST_EPISODES} test episodes across all 4 controllers...")

    print("\n" + "=" * 78)
    print("FOUR-WAY BENCHMARK EMPIRICAL RESULTS (100 SEVERE DISTURBANCE EPISODES)")
    print("=" * 78)
    
    summary_data = {}
    for c_name in controllers:
        stat = benchmark_stats[c_name]
        rec_rate = (stat["success_count"] / N_TEST_EPISODES) * 100.0
        tum_rate = (stat["tumble_count"] / N_TEST_EPISODES) * 100.0
        viol_rate = (stat["constraint_violations"] / max(1, stat["total_steps"])) * 100.0
        avg_energy = stat["total_energy"] / N_TEST_EPISODES
        avg_latency = float(np.mean(stat["latencies_ms"]))
        
        summary_data[c_name] = {
            "recovery_success_rate": rec_rate,
            "tumble_rate": tum_rate,
            "constraint_violation_rate": viol_rate,
            "avg_control_energy": avg_energy,
            "latency_ms": avg_latency
        }
        
        print(f"• {c_name:38s}:")
        print(f"    - Recovery Success Rate:    {rec_rate:6.1f} %")
        print(f"    - Tumble / Fall Rate:       {tum_rate:6.1f} %")
        print(f"    - Invariant Violation Rate: {viol_rate:6.1f} %")
        print(f"    - Mean Control Energy:      {avg_energy:6.2f} (N*m)^2*s")
        print(f"    - Mean Inference Latency:   {avg_latency:6.3f} ms")
    print("=" * 78)
    
    with open("/Volumes/KIOXIA/difusionMPC/checkpoints/four_way_benchmark_results.json", "w") as f:
        json.dump(summary_data, f, indent=2)
        
    # Plotting comparison figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    names_abbr = ["Pure MPC", "Det. BC", "Pure CFM", "Ours\n(CFM+CBF)"]
    colors = ['#1f77b4', '#d62728', '#ff7f0e', '#2ca02c']
    
    # 1. Recovery Rate
    rec_vals = [summary_data[c]["recovery_success_rate"] for c in controllers]
    axes[0].bar(names_abbr, rec_vals, color=colors, alpha=0.85, edgecolor='black')
    axes[0].set_title("Recovery Success Rate (%) ↑", fontsize=12, fontweight='bold')
    axes[0].set_ylim(0, 105)
    axes[0].grid(axis='y', alpha=0.3)
    for i, v in enumerate(rec_vals):
        axes[0].text(i, v + 2, f"{v:.1f}%", ha='center', fontweight='bold')
        
    # 2. Invariant Violation Rate
    viol_vals = [summary_data[c]["constraint_violation_rate"] for c in controllers]
    axes[1].bar(names_abbr, viol_vals, color=colors, alpha=0.85, edgecolor='black')
    axes[1].set_title("Invariant Violation Rate (%) ↓", fontsize=12, fontweight='bold')
    axes[1].set_ylim(0, max(viol_vals) * 1.25 if max(viol_vals) > 0 else 10)
    axes[1].grid(axis='y', alpha=0.3)
    for i, v in enumerate(viol_vals):
        axes[1].text(i, v + 1, f"{v:.1f}%", ha='center', fontweight='bold')
        
    # 3. Latency vs Energy Tradeoff
    lat_vals = [summary_data[c]["latency_ms"] for c in controllers]
    axes[2].bar(names_abbr, lat_vals, color=colors, alpha=0.85, edgecolor='black')
    axes[2].set_title("Inference Latency per Step (ms) ↓", fontsize=12, fontweight='bold')
    axes[2].grid(axis='y', alpha=0.3)
    for i, v in enumerate(lat_vals):
        axes[2].text(i, v + 0.05 * max(lat_vals), f"{v:.2f}ms", ha='center', fontweight='bold')
        
    plt.tight_layout()
    chart_path = "/Volumes/KIOXIA/difusionMPC/four_way_benchmark_comparison.png"
    plt.savefig(chart_path, dpi=300)
    plt.close()
    print(f"\nSaved benchmark comparison chart to {chart_path}")

if __name__ == "__main__":
    generate_dataset_and_train()
    run_four_way_benchmark()
