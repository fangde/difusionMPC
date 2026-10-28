import time
import json
import numpy as np
import scipy.linalg
from scipy.optimize import lsq_linear
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

# ==============================================================================
# HIERARCHICAL CONTROL VALIDATION:
# POLICY PLANS FEASIBLE TRAJECTORY -> PLANT CALCULATES TRACKING TORQUE
#
# Double-checking the core architectural insight:
# 1. High-Level Policy learns from the MPC recovery vector field:
#    It outputs the dynamically feasible state evolution / reference trajectory:
#    x_t -> Policy -> x_ref,t+1 (or v_ref)
# 2. Plant executes Low-Level Tracking Control:
#    u_plant = PlantInverseDynamics(x_t, x_ref,t+1)
# 3. Model Perturbation Stress Test:
#    The Plant has perturbed physical parameters:
#    m_plant != m_nominal, l_plant != l_nominal, b_plant > 0 (unmodeled friction).
# 4. Feasibility Criterion:
#    - If |u_plant| <= u_max: The planned trajectory is FEASIBLE for the perturbed plant.
#    - If |u_plant| > u_max: Actuator saturates -> Path INFEASIBLE under plant change!
# ==============================================================================

np.random.seed(42)
torch.manual_seed(42)

# Nominal Model (used by MPC to construct reference vector field)
m_nom, l_nom = 1.0, 1.0
g, dt, u_max = 9.81, 0.02, 12.0
I_nom = m_nom * l_nom**2

# Nominal Discretized MPC
A_c = np.array([[0.0, 1.0], [g / l_nom, 0.0]])
B_c = np.array([[0.0], [1.0 / I_nom]])
Q_cost = np.diag([20.0, 2.0])
R_cost = np.array([[0.5]])
A_d = scipy.linalg.expm(A_c * dt)
B_d = np.linalg.inv(A_c) @ (A_d - np.eye(2)) @ B_c
P_dare = scipy.linalg.solve_discrete_are(A_d, B_d, Q_cost, R_cost)

def solve_nominal_mpc(s, H=10):
    nx, nu = 2, 1
    Phi = np.zeros((H * nx, nx))
    Gamma = np.zeros((H * nx, H * nu))
    A_p = np.eye(nx)
    for i in range(H):
        A_p = A_p @ A_d
        Phi[i*nx:(i+1)*nx, :] = A_p
        for j in range(i + 1):
            Gamma[i*nx:(i+1)*nx, j*nu:(j+1)*nu] = np.linalg.matrix_power(A_d, i - j) @ B_d
    Q_bar = np.kron(np.eye(H), Q_cost)
    Q_bar[(H-1)*nx:, (H-1)*nx:] = P_dare
    R_bar = np.kron(np.eye(H), R_cost)
    Hess = 2.0 * (R_bar + Gamma.T @ Q_bar @ Gamma)
    grad = 2.0 * (Gamma.T @ Q_bar @ Phi @ s)
    L = np.linalg.cholesky(Hess)
    d = -np.linalg.solve(L, grad)
    res = lsq_linear(L.T, d, bounds=(-u_max, u_max), verbose=0)
    return float(res.x[0])

# ------------------------------------------------------------------------------
# 1. Dataset Generation: State Transitions x_t -> x_ref,t+1 (Feasible Trajectories)
# ------------------------------------------------------------------------------
print("[Step 1/4] Generating Feasible State Trajectory Dataset from Nominal MPC...")
N_SAMPLES = 12000
states_train = []
next_states_ref = []

for _ in range(N_SAMPLES):
    th = np.random.uniform(-0.55, 0.55)
    dth = np.random.uniform(-1.20, 1.20)
    s_t = np.array([th, dth])
    
    # Nominal MPC action
    u_nom = solve_local_mpc(s_t) if 'solve_local_mpc' in locals() else solve_nominal_mpc(s_t)
    
    # Nominal transition using exact RK4
    def deriv_nom(s, act): return np.array([s[1], (g / l_nom) * np.sin(s[0]) + act / I_nom])
    k1 = deriv_nom(s_t, u_nom)
    k2 = deriv_nom(s_t + 0.5 * dt * k1, u_nom)
    k3 = deriv_nom(s_t + 0.5 * dt * k2, u_nom)
    k4 = deriv_nom(s_t + dt * k3, u_nom)
    s_next = s_t + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
    s_next[0] = (s_next[0] + np.pi) % (2 * np.pi) - np.pi
    
    states_train.append(s_t)
    next_states_ref.append(s_next)

X_train = np.array(states_train, dtype=np.float32)
Y_ref = np.array(next_states_ref, dtype=np.float32)
print(f"  -> Generated {len(X_train)} feasible state transition pairs (x_t -> x_ref,t+1)")

# ------------------------------------------------------------------------------
# 2. Train High-Level Trajectory Planning Policy
#    x_t -> Policy -> x_ref,t+1
# ------------------------------------------------------------------------------
print("\n[Step 2/4] Training Generative Trajectory Planning Policy (Flow Matching)...")

class TrajectoryFlowNet(nn.Module):
    """Predicts feasible next reference state x_ref from current state x_t"""
    def __init__(self, state_dim=2, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + state_dim + 1, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, state_dim)
        )
    def forward(self, x_cand, x_t, t):
        return self.net(torch.cat([x_cand, x_t, t], dim=-1))

loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(torch.tensor(X_train), torch.tensor(Y_ref)),
    batch_size=128, shuffle=True
)

policy_flow = TrajectoryFlowNet()
opt = torch.optim.Adam(policy_flow.parameters(), lr=1.5e-3)

for epoch in range(20):
    for s_t, s_target in loader:
        opt.zero_grad()
        B = s_t.shape[0]
        t = torch.rand(B, 1)
        s_0 = torch.randn_like(s_target) * 0.1 + s_t # noise around current state
        s_tau = (1.0 - t) * s_0 + t * s_target
        v_target = s_target - s_0
        pred_v = policy_flow(s_tau, s_t, t)
        loss = F.mse_loss(pred_v, v_target)
        loss.backward()
        opt.step()

print("  -> Policy trained to generate feasible state trajectory predictions.")

def plan_next_reference_state(current_state_np, num_steps=5):
    with torch.no_grad():
        s_curr = torch.tensor(current_state_np, dtype=torch.float32).unsqueeze(0)
        s_gen = s_curr.clone() # initialize at current state
        dt_s = 1.0 / num_steps
        for i in range(num_steps):
            t_val = torch.tensor([[i * dt_s]], dtype=torch.float32)
            v = policy_flow(s_gen, s_curr, t_val)
            s_gen = s_gen + v * dt_s
        return s_gen.numpy()[0]

# ------------------------------------------------------------------------------
# 3. Low-Level Plant Tracking Controller under Model Perturbations
# ------------------------------------------------------------------------------
class PerturbedPlant:
    """
    Physical hardware with real perturbed parameters:
    m_real = m_nom * (1 + delta_m)
    l_real = l_nom * (1 + delta_l)
    b_real = joint damping friction
    """
    def __init__(self, delta_m=0.30, delta_l=0.20, b_friction=0.20):
        self.m = m_nom * (1.0 + delta_m)
        self.l = l_nom * (1.0 + delta_l)
        self.b = b_friction
        self.I = self.m * self.l**2
        self.u_max = u_max

        # Tracking Controller Gains (PD feedback on tracking error)
        self.Kp = 45.0
        self.Kd = 12.0

    def compute_tracking_torque(self, current_state, reference_state):
        th_curr, dth_curr = current_state
        th_ref, dth_ref = reference_state
        
        # Desired acceleration to follow planned reference state
        # e = th_ref - th_curr, de = dth_ref - dth_curr
        err_th = ((th_ref - th_curr + np.pi) % (2*np.pi)) - np.pi
        err_dth = dth_ref - dth_curr
        
        accel_des = (err_dth / dt) + self.Kp * err_th + self.Kd * err_dth
        
        # Inverse Dynamics computed by Plant tracking controller:
        # u_plant = I * (accel_des - (g/l)*sin(theta)) + b * omega
        u_plant = self.I * (accel_des - (g / self.l) * np.sin(th_curr)) + self.b * dth_curr
        
        # Physical saturation check
        is_feasible = abs(u_plant) <= self.u_max
        u_actual = float(np.clip(u_plant, -self.u_max, self.u_max))
        return u_actual, is_feasible, u_plant

    def step(self, state, u_torque):
        th, dth = state
        u_eff = float(np.clip(u_torque, -self.u_max, self.u_max))
        def deriv(s, act):
            return np.array([s[1], (g / self.l) * np.sin(s[0]) + act / self.I - (self.b / self.I) * s[1]])
        k1 = deriv(state, u_eff)
        k2 = deriv(state + 0.5 * dt * k1, u_eff)
        k3 = deriv(state + 0.5 * dt * k2, u_eff)
        k4 = deriv(state + dt * k3, u_eff)
        ns = state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        ns[0] = (ns[0] + np.pi) % (2*np.pi) - np.pi
        return ns

# ------------------------------------------------------------------------------
# 4. Stress-Testing Model Perturbations across Parameter Grid
# ------------------------------------------------------------------------------
print("\n[Step 3/4] Stress-Testing Closed-Loop Stability under Model Perturbations...")

# Parameter Perturbation Matrix:
# delta_m in [-30%, 0%, +30%, +50%]
# delta_l in [-20%, 0%, +20%, +35%]
# b_friction in [0.0, 0.15, 0.35]
perturbation_test_cases = [
    ("Nominal Plant (0% mismatch)", 0.0, 0.0, 0.0),
    ("Moderate Plant (+20% mass, +15% length, b=0.1)", 0.20, 0.15, 0.10),
    ("Heavy Plant (+40% mass, +25% length, b=0.25)", 0.40, 0.25, 0.25),
    ("Extreme Plant (+60% mass, +35% length, b=0.35)", 0.60, 0.35, 0.35),
    ("Light Fast Plant (-25% mass, -15% length, b=0.05)", -0.25, -0.15, 0.05),
]

N_TRIALS = 50
SIM_STEPS = 120 # 2.4 seconds

benchmark_report = {}

for case_name, dm, dl, b_fric in perturbation_test_cases:
    plant = PerturbedPlant(delta_m=dm, delta_l=dl, b_friction=b_fric)
    
    succ_count = 0
    feasible_steps_total = 0
    total_steps = 0
    peak_torques = []
    tracking_errors = []
    
    for trial in range(N_TRIALS):
        np.random.seed(3000 + trial)
        th0 = np.random.uniform(-0.40, 0.40)
        dth0 = np.random.uniform(-0.40, 0.40)
        s_curr = np.array([th0, dth0])
        
        fell = False
        trial_errors = []
        
        for step in range(SIM_STEPS):
            # 1. High-level Policy plans feasible reference state
            x_ref = plan_next_reference_state(s_curr)
            
            # 2. Low-level Plant computes tracking torque u_plant
            u_actual, is_feasible, u_raw = plant.compute_tracking_torque(s_curr, x_ref)
            
            total_steps += 1
            if is_feasible:
                feasible_steps_total += 1
            peak_torques.append(abs(u_raw))
            
            # Step physical plant
            s_next = plant.step(s_curr, u_actual)
            
            # Tracking error: difference between actual state and planned reference
            err = np.linalg.norm(s_next - x_ref)
            trial_errors.append(err)
            
            s_curr = s_next
            if abs(s_curr[0]) > 0.8: # Tumbled
                fell = True
                break
                
        tracking_errors.append(np.mean(trial_errors))
        if not fell and abs(s_curr[0]) < 0.06 and abs(s_curr[1]) < 0.15:
            succ_count += 1
            
    recovery_rate = (succ_count / N_TRIALS) * 100.0
    feasibility_rate = (feasible_steps_total / max(1, total_steps)) * 100.0
    mean_err = np.mean(tracking_errors)
    mean_peak_u = np.mean(peak_torques)
    
    benchmark_report[case_name] = {
        "recovery_rate": recovery_rate,
        "feasibility_rate": feasibility_rate,
        "tracking_error": mean_err,
        "mean_torque": mean_peak_u
    }

print("\n" + "=" * 80)
print("HIERARCHICAL CONTROL RESULTS UNDER PHYSICAL MODEL PERTURBATION:")
print("=" * 80)
print(f"{'Plant Condition':<48} | {'Recovery Rate':<13} | {'Path Feasibility':<16} | {'Tracking Error':<14}")
print("-" * 80)
for name, data in benchmark_report.items():
    print(f"• {name:<46} | {data['recovery_rate']:5.1f}%       | {data['feasibility_rate']:5.1f}%          | {data['tracking_error']:6.4f}")
print("=" * 80)

# ------------------------------------------------------------------------------
# 5. Publication Figure: Hierarchical Architecture & Robustness Validation
# ------------------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(16, 5), dpi=300)

cases_abbr = ["Nominal\n(0%)", "Moderate\n(+20%m, +15%l)", "Heavy\n(+40%m, +25%l)", "Extreme\n(+60%m, +35%l)", "Light\n(-25%m, -15%l)"]
colors = ['#1f77b4', '#2ca02c', '#10b981', '#f59e0b', '#8b5cf6']

# 1. Recovery Success Rate (%)
rec_vals = [benchmark_report[k]["recovery_rate"] for k in benchmark_report]
axes[0].bar(cases_abbr, rec_vals, color=colors, alpha=0.85, edgecolor='black')
axes[0].set_title("(A) Closed-Loop Recovery Success Rate ↑", fontsize=11, fontweight='bold')
axes[0].set_ylabel("Success Rate (%)", fontweight='bold')
axes[0].set_ylim(0, 115)
axes[0].grid(axis='y', alpha=0.3)
for i, v in enumerate(rec_vals):
    axes[0].text(i, v + 2, f"{v:.1f}%", ha='center', fontweight='bold', fontsize=9)

# 2. Path Dynamic Feasibility Rate (%) (|u_plant| <= u_max)
feas_vals = [benchmark_report[k]["feasibility_rate"] for k in benchmark_report]
axes[1].bar(cases_abbr, feas_vals, color=colors, alpha=0.85, edgecolor='black')
axes[1].set_title("(B) Path Feasibility Rate (|u_plant| ≤ u_max) ↑", fontsize=11, fontweight='bold')
axes[1].set_ylabel("Feasibility Rate (%)", fontweight='bold')
axes[1].set_ylim(0, 115)
axes[1].grid(axis='y', alpha=0.3)
for i, v in enumerate(feas_vals):
    axes[1].text(i, v + 2, f"{v:.1f}%", ha='center', fontweight='bold', fontsize=9)

# 3. Tracking Error between Planned Path and Real Plant
err_vals = [benchmark_report[k]["tracking_error"] for k in benchmark_report]
axes[2].bar(cases_abbr, err_vals, color=colors, alpha=0.85, edgecolor='black')
axes[2].set_title("(C) Trajectory Tracking Error ||x_plant - x_ref|| ↓", fontsize=11, fontweight='bold')
axes[2].set_ylabel("Mean Tracking Error", fontweight='bold')
axes[2].grid(axis='y', alpha=0.3)
for i, v in enumerate(err_vals):
    axes[2].text(i, v + 0.003, f"{v:.3f}", ha='center', fontweight='bold', fontsize=9)

plt.tight_layout()
out_plot = "figures/hierarchical_tracking_model_perturbation.png"
plt.savefig(out_plot, dpi=300)
plt.close()
print(f"\n[Step 4/4] Successfully saved hierarchical tracking benchmark figure to {out_plot}")
