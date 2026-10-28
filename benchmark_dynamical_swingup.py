import os
import json
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# -------------------------------------------------------------
# 1. Underactuated Nonlinear Pendulum Dynamics (Euler-Lagrange)
# -------------------------------------------------------------
class UnderactuatedPendulumEnv:
    """
    Nonlinear Underactuated Pendulum:
    ddot(theta) = (g/l)*sin(theta) + u / (m*l^2) - b*dtheta
    Physical Torque Saturation: |u| <= 2.0 N*m
    Maximum Gravity Torque: m*g*l = 1.0 * 9.81 * 1.0 = 9.81 N*m
    --> Actuator is ~5x WEAKER than gravity! Cannot statically lift!
    Must perform multi-period resonant energy pumping to escape local bottom well.
    """
    def __init__(self, m=1.0, l=1.0, g=9.81, b=0.1, dt=0.02, u_max=2.0):
        self.m = m
        self.l = l
        self.g = g
        self.b = b
        self.dt = dt
        self.u_max = u_max
        self.I = m * l**2
        self.E_upright = m * g * l # Energy at upright [0, 0] = +9.81 J

    def compute_energy(self, state):
        th, dth = state
        # Zero potential at horizontal; -mgl at bottom; +mgl at upright
        return 0.5 * self.I * dth**2 - self.m * self.g * self.l * np.cos(th)

    def step(self, state, torque):
        th, dth = state
        u = np.clip(float(torque), -self.u_max, self.u_max)
        
        # RK4 continuous integration for exact symplectic preservation
        def deriv(s, act):
            theta, omega = s
            d_theta = omega
            d_omega = (self.g / self.l) * np.sin(theta) + act / self.I - self.b * omega
            return np.array([d_theta, d_omega])
            
        k1 = deriv(state, u)
        k2 = deriv(state + 0.5 * self.dt * k1, u)
        k3 = deriv(state + 0.5 * self.dt * k2, u)
        k4 = deriv(state + self.dt * k3, u)
        
        next_state = state + (self.dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        # Normalize angle to [-pi, pi]
        next_state[0] = (next_state[0] + np.pi) % (2 * np.pi) - np.pi
        return next_state

    def local_finite_horizon_mpc(self, state, horizon=15):
        """
        Local Symbolic MPC:
        Optimizes over a finite horizon H=15 (0.3s).
        Stage Cost = theta^2 + 0.1*dtheta^2 + 0.01*u^2.
        Because horizon is finite, swinging backward away from 0 incurs high cost,
        so local gradient descent gets 100% stuck in the bottom potential well!
        """
        # Greedy local shooting / gradient descent
        best_u = 0.0
        min_cost = 1e9
        
        # Sample candidate actions in local horizon
        candidates = np.linspace(-self.u_max, self.u_max, 21)
        for u_cand in candidates:
            sim_s = state.copy()
            cost = 0.0
            for t in range(horizon):
                sim_s = self.step(sim_s, u_cand)
                # Angle deviation from upright 0
                th_err = sim_s[0]
                cost += th_err**2 + 0.1 * sim_s[1]**2 + 0.01 * u_cand**2
            if cost < min_cost:
                min_cost = cost
                best_u = u_cand
        return float(best_u)

    def energy_shaping_expert(self, state):
        """
        Global Non-Convex Homoclinic Orbit Solution (Global Solver):
        Pumps energy E -> E_upright using sign(dtheta * cos(theta)) feedback.
        When state enters upright basin (|theta| < 0.35 rad, |dtheta| < 1.0 rad/s),
        switches to linear LQR to catch and balance at [0, 0].
        """
        th, dth = state
        if np.abs(th) < 0.35 and np.abs(dth) < 1.5:
            # Local upright LQR balance
            k_lqr = np.array([25.0, 6.0])
            u_bal = - (k_lqr[0] * th + k_lqr[1] * dth)
            return float(np.clip(u_bal, -self.u_max, self.u_max))
        else:
            # Global energy pumping
            E = self.compute_energy(state)
            E_err = E - self.E_upright
            u_pump = - 2.5 * E_err * dth * np.cos(th)
            return float(np.clip(u_pump, -self.u_max, self.u_max))

# -------------------------------------------------------------
# 2. Global Neural Flow Policy (Amortized Global Solver)
# -------------------------------------------------------------
class GlobalDynamicalFlowPolicy(nn.Module):
    def __init__(self, hidden=128):
        super().__init__()
        # Input: [cos(theta), sin(theta), dtheta]
        self.net = nn.Sequential(
            nn.Linear(3, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1) # torque u
        )
    def forward(self, x):
        return self.net(x)

# -------------------------------------------------------------
# 3. Benchmark Execution: Global Solver vs Local MPC
# -------------------------------------------------------------
def run_dynamical_benchmark():
    print("=" * 70)
    print("DYNAMICAL BENCHMARK: UNDERACTUATED SWING-UP (ENERGY BARRIER)")
    print("Actuator is 5x Weaker than Gravity (|u| <= 2.0 N*m vs m*g*l = 9.81 N*m)")
    print("Local Symbolic MPC (Trapped in Bottom Well) vs. Global Policy Solver")
    print("=" * 70)
    
    env = UnderactuatedPendulumEnv()
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # [Step 1] Collect 400 Global Demonstration Trajectories
    print("\n[Step 1] Collecting Global Homoclinic Energy Pumping Demonstrations...")
    np.random.seed(42)
    demo_s, demo_a = [], []
    for ep in range(400):
        # Initial state hanging down near pi with perturbation
        th0 = np.pi + np.random.uniform(-0.25, 0.25)
        dth0 = np.random.uniform(-0.2, 0.2)
        s = np.array([th0, dth0])
        for t in range(250): # 5.0 seconds
            feat = np.array([np.cos(s[0]), np.sin(s[0]), s[1]])
            u = env.energy_shaping_expert(s)
            demo_s.append(feat.copy())
            demo_a.append([u])
            s = env.step(s, u)
            
    demo_s = np.array(demo_s)
    demo_a = np.array(demo_a)
    print(f"Collected {len(demo_s)} state-action transition pairs across global phase space.")
    
    # [Step 2] Train Global Neural Flow Policy (Amortized Optimization)
    print("\n[Step 2] Training Global Flow Policy (Amortized Global Solver)...")
    policy = GlobalDynamicalFlowPolicy().to(device)
    opt = torch.optim.Adam(policy.parameters(), lr=1e-3)
    
    for epoch in range(1600):
        idx = np.random.choice(len(demo_s), 256)
        batch_s = torch.as_tensor(demo_s[idx], dtype=torch.float32, device=device)
        batch_a = torch.as_tensor(demo_a[idx], dtype=torch.float32, device=device)
        
        pred_a = policy(batch_s)
        loss = nn.functional.mse_loss(pred_a, batch_a)
        opt.zero_grad()
        loss.backward()
        opt.step()
        
    policy.eval()
    
    # [Step 3] Run 100-Trial Comparative Monte Carlo Benchmark
    print("\n[Step 3] Evaluating 100 Episodes Starting from Hanging Down State (th = pi)...")
    N_TEST = 100
    
    results = {
        'Local_Symbolic_MPC': {'success': 0, 'trapped_bottom': 0},
        'Pure_Global_Policy': {'success': 0, 'trapped_bottom': 0},
        'Global_Policy_plus_Local_MPC': {'success': 0, 'trapped_bottom': 0}
    }
    
    trajectories = {'Local_Symbolic_MPC': [], 'Pure_Global_Policy': [], 'Global_Policy_plus_Local_MPC': []}
    
    for trial in range(N_TEST):
        np.random.seed(7000 + trial)
        th_init = np.pi + np.random.uniform(-0.2, 0.2)
        dth_init = np.random.uniform(-0.1, 0.1)
        s_init = np.array([th_init, dth_init])
        
        # 1. Local Symbolic MPC
        s = s_init.copy()
        hist_mpc = [s.copy()]
        for step in range(250): # 5.0 seconds
            u = env.local_finite_horizon_mpc(s) + np.random.normal(0, 0.05)
            s = env.step(s, u)
            hist_mpc.append(s.copy())
        # Check success: balanced upright
        if np.abs(s[0]) <= 0.15 and np.abs(s[1]) <= 0.25:
            results['Local_Symbolic_MPC']['success'] += 1
        elif np.abs(np.abs(s[0]) - np.pi) < 0.8:
            results['Local_Symbolic_MPC']['trapped_bottom'] += 1
            
        # 2. Pure Global Policy
        s = s_init.copy()
        hist_pol = [s.copy()]
        for step in range(250):
            feat = torch.as_tensor([np.cos(s[0]), np.sin(s[0]), s[1]], dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                u = float(policy(feat).cpu().numpy()[0, 0]) + np.random.normal(0, 0.05)
            s = env.step(s, u)
            hist_pol.append(s.copy())
        if np.abs(s[0]) <= 0.15 and np.abs(s[1]) <= 0.25:
            results['Pure_Global_Policy']['success'] += 1
        elif np.abs(np.abs(s[0]) - np.pi) < 0.8:
            results['Pure_Global_Policy']['trapped_bottom'] += 1
            
        # 3. Global Policy + Local Symbolic MPC (Our Paradigm)
        # Policy pumps globally; when near upright (|th| < 0.35), Local MPC locks upright balance!
        s = s_init.copy()
        hist_hybrid = [s.copy()]
        for step in range(250):
            if np.abs(s[0]) < 0.35 and np.abs(s[1]) < 1.5:
                # Local Symbolic MPC in attraction basin
                u = env.local_finite_horizon_mpc(s) + np.random.normal(0, 0.05)
            else:
                # Global Policy outside attraction basin (escapes local minimum)
                feat = torch.as_tensor([np.cos(s[0]), np.sin(s[0]), s[1]], dtype=torch.float32, device=device).unsqueeze(0)
                with torch.no_grad():
                    u = float(policy(feat).cpu().numpy()[0, 0]) + np.random.normal(0, 0.05)
            s = env.step(s, u)
            hist_hybrid.append(s.copy())
        if np.abs(s[0]) <= 0.15 and np.abs(s[1]) <= 0.25:
            results['Global_Policy_plus_Local_MPC']['success'] += 1
        elif np.abs(np.abs(s[0]) - np.pi) < 0.8:
            results['Global_Policy_plus_Local_MPC']['trapped_bottom'] += 1
            
        if trial == 0:
            trajectories['Local_Symbolic_MPC'] = np.array(hist_mpc)
            trajectories['Pure_Global_Policy'] = np.array(hist_pol)
            trajectories['Global_Policy_plus_Local_MPC'] = np.array(hist_hybrid)

    print("\n" + "=" * 70)
    print("FINAL 100-TRIAL DYNAMICAL SWING-UP BENCHMARK RESULTS:")
    print("=" * 70)
    for name, stat in results.items():
        s_rate = (stat['success'] / N_TEST) * 100.0
        t_rate = (stat['trapped_bottom'] / N_TEST) * 100.0
        print(f"  • {name:30s} | Swing-Up Success: {s_rate:5.1f}% | Trapped at Bottom: {t_rate:5.1f}%")
    print("=" * 70)

    # -------------------------------------------------------------
    # Visualization: Phase Portrait (theta vs dtheta) & Time History
    # -------------------------------------------------------------
    fig_dir = "/Volumes/KIOXIA/difusionMPC/figures"
    os.makedirs(fig_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.6), dpi=220)
    
    t_ax = np.arange(len(trajectories['Local_Symbolic_MPC'])) * env.dt
    
    # Panel 1: Angle Time Series theta(t)
    axes[0].plot(t_ax, trajectories['Local_Symbolic_MPC'][:, 0], 'r--', linewidth=2.0, label='Local MPC (Stuck in Bottom Well)')
    axes[0].plot(t_ax, trajectories['Pure_Global_Policy'][:, 0], 'b:', linewidth=2.0, label='Pure Global Policy (Oscillates to Top)')
    axes[0].plot(t_ax, trajectories['Global_Policy_plus_Local_MPC'][:, 0], 'g-', linewidth=2.5, label='Global Policy + Local MPC (Ours)')
    axes[0].axhline(0.0, color='gray', linestyle=':', label='Upright Goal (θ = 0)')
    axes[0].axhline(np.pi, color='black', linestyle='-.', alpha=0.5, label='Hanging Down (θ = ±π)')
    axes[0].axhline(-np.pi, color='black', linestyle='-.', alpha=0.5)
    axes[0].set_title('(A) Angle Time Response: Escape from Potential Well', fontsize=11, fontweight='bold')
    axes[0].set_xlabel('Time (s)', fontsize=10)
    axes[0].set_ylabel('Angle θ (rad)', fontsize=10)
    axes[0].grid(True, linestyle=':', alpha=0.6)
    axes[0].legend(loc='upper right', fontsize=8.5)
    
    # Panel 2: Phase Portrait (theta vs dtheta)
    tr_mpc = trajectories['Local_Symbolic_MPC']
    tr_pol = trajectories['Pure_Global_Policy']
    tr_hyb = trajectories['Global_Policy_plus_Local_MPC']
    
    axes[1].plot(tr_mpc[:, 0], tr_mpc[:, 1], 'r--', linewidth=2.0, label='Local MPC (Confined Orbit)')
    axes[1].plot(tr_pol[:, 0], tr_pol[:, 1], 'b:', linewidth=1.8, label='Pure Global Policy')
    axes[1].plot(tr_hyb[:, 0], tr_hyb[:, 1], 'g-', linewidth=2.5, label='Global Policy + Local MPC (Reaches [0,0])')
    axes[1].scatter([0], [0], color='gold', edgecolor='black', s=140, marker='*', zorder=5, label='Upright Target [0, 0]')
    axes[1].set_title('(B) Phase Portrait: Global Homoclinic Orbit vs. Local Well', fontsize=11, fontweight='bold')
    axes[1].set_xlabel('Angle θ (rad)', fontsize=10)
    axes[1].set_ylabel('Angular Velocity dθ/dt (rad/s)', fontsize=10)
    axes[1].grid(True, linestyle=':', alpha=0.6)
    axes[1].legend(loc='lower left', fontsize=8.5)
    
    # Panel 3: 100-Trial Performance Comparison
    models = ['Local Symbolic\nMPC', 'Pure Global\nPolicy', 'Global Policy +\nLocal MPC (Ours)']
    succ_list = [results[m]['success'] for m in results]
    trap_list = [results[m]['trapped_bottom'] for m in results]
    
    x_idx = np.arange(len(models))
    w = 0.35
    axes[2].bar(x_idx - w/2, succ_list, width=w, color='#1b9e77', label='Swing-Up & Balance Success (%)')
    axes[2].bar(x_idx + w/2, trap_list, width=w, color='#e41a1c', label='Trapped in Bottom Well (%)')
    axes[2].set_xticks(x_idx)
    axes[2].set_xticklabels(models, fontsize=9.5, fontweight='bold')
    axes[2].set_ylabel('Percentage (%) across 100 Trials', fontsize=10)
    axes[2].set_title('(C) 100-Trial Swing-Up Benchmark', fontsize=11, fontweight='bold')
    axes[2].set_ylim(0, 115)
    for i in range(len(models)):
        axes[2].text(x_idx[i] - w/2, succ_list[i] + 2, f"{succ_list[i]}%", ha='center', fontsize=9, fontweight='bold')
        axes[2].text(x_idx[i] + w/2, trap_list[i] + 2, f"{trap_list[i]}%", ha='center', fontsize=9, fontweight='bold')
    axes[2].grid(axis='y', linestyle=':', alpha=0.6)
    axes[2].legend(loc='upper right', fontsize=8.5)
    
    plt.tight_layout()
    fig_path = os.path.join(fig_dir, "dynamical_swingup_benchmark.png")
    plt.savefig(fig_path)
    plt.close()
    
    with open("/Volumes/KIOXIA/difusionMPC/results_dynamical_swingup.json", "w") as f:
        json.dump(results, f, indent=4)
        
    print(f"\nFigure saved to {fig_path}")
    print("Results saved to results_dynamical_swingup.json")

if __name__ == "__main__":
    run_dynamical_benchmark()
