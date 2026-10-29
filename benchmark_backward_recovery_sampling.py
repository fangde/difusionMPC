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
# BENCHMARK: BACKWARD RECOVERY SAMPLING (BRS) & ACTIVE BASIN EXPANSION
#
# Comparing:
#   Group A: Uniform Random Sampling + MPC
#   Group B: Reverse Random Walk from Goal (Florensa style) + MPC
#   Group C: Backward Recovery Sampling with Active Boundary Frontier (BRS) + MPC
#   Group D: Policy Distillation onto Deterministic BC (trained on C's data)
#   Group E: Policy Distillation onto Generative Recovery Diffusion (MGRD, trained on C's data)
#
# Metrics Evaluated:
#   1. Basin Coverage Ratio (% of Ground Truth Viable Basin B*)
#   2. MPC Yield Efficiency (Valid Recovery States per MPC Query)
#   3. Active Boundary Growth Rate (Basin Volume per Compute Unit)
#   4. Closed-Loop Dynamic Stabilization Rate under Perturbations
# ==============================================================================

np.random.seed(42)
torch.manual_seed(42)
device = torch.device('cpu')

# ------------------------------------------------------------------------------
# 1. Physical System: Torque-Saturated Inverted Pendulum
#    State: x = [theta, omega]^T,  Action: u in [-u_max, u_max]
#    Goal:  x* = [0, 0]^T
# ------------------------------------------------------------------------------
m, l, g, dt = 1.0, 1.0, 9.81, 0.02
I = m * l**2
u_max = 8.5  # Saturation creates a non-convex, curved Basin boundary!

def deriv(state, act):
    act = float(np.clip(act, -u_max, u_max))
    return np.array([state[1], (g / l) * np.sin(state[0]) + act / I])

def step_forward(state, act):
    k1 = deriv(state, act)
    k2 = deriv(state + 0.5 * dt * k1, act)
    k3 = deriv(state + 0.5 * dt * k2, act)
    k4 = deriv(state + dt * k3, act)
    ns = state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    ns[0] = (ns[0] + np.pi) % (2.0 * np.pi) - np.pi
    return ns

def step_backward(state, act):
    """
    Inverse dynamic predecessor step x^-:
    Reverses time (dt -> -dt) under action act.
    f(x^-, act) approx x.
    """
    k1 = deriv(state, act)
    k2 = deriv(state - 0.5 * dt * k1, act)
    k3 = deriv(state - 0.5 * dt * k2, act)
    k4 = deriv(state - dt * k3, act)
    prev_s = state - (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    prev_s[0] = (prev_s[0] + np.pi) % (2.0 * np.pi) - np.pi
    return prev_s

# MPC Setup
A_c = np.array([[0.0, 1.0], [g / l, 0.0]])
B_c = np.array([[0.0], [1.0 / I]])
Q_cost = np.diag([25.0, 2.5])
R_cost = np.array([[0.4]])
A_d = scipy.linalg.expm(A_c * dt)
B_d = np.linalg.inv(A_c) @ (A_d - np.eye(2)) @ B_c
P_dare = scipy.linalg.solve_discrete_are(A_d, B_d, Q_cost, R_cost)

H = 12
nx, nu = 2, 1
Phi = np.zeros((H * nx, nx))
Gamma = np.zeros((H * nx, H * nu))
A_p = np.eye(nx)
for i in range(H):
    A_p = A_p @ A_d
    Phi[i * nx:(i + 1) * nx, :] = A_p
    for j in range(i + 1):
        Gamma[i * nx:(i + 1) * nx, j * nu:(j + 1) * nu] = np.linalg.matrix_power(A_d, i - j) @ B_d

Q_bar = np.kron(np.eye(H), Q_cost)
Q_bar[(H - 1) * nx:, (H - 1) * nx:] = P_dare
R_bar = np.kron(np.eye(H), R_cost)
Hess = 2.0 * (R_bar + Gamma.T @ Q_bar @ Gamma)
L_chol = np.linalg.cholesky(Hess)

def solve_mpc(s):
    grad_qp = 2.0 * (Gamma.T @ Q_bar @ Phi @ s)
    d = -np.linalg.solve(L_chol, grad_qp)
    res = lsq_linear(L_chol.T, d, bounds=(-u_max, u_max), verbose=0)
    return float(res.x[0])

def check_mpc_recovery(start_state, max_steps=25):
    """Verifies whether online MPC safely contracts state to goal |x| < 0.1"""
    curr = start_state.copy()
    traj = [curr.copy()]
    acts = []
    for _ in range(max_steps):
        u = solve_mpc(curr)
        curr = step_forward(curr, u)
        traj.append(curr.copy())
        acts.append(u)
        if np.abs(curr[0]) > np.pi * 0.8: # toppled over
            return False, traj, acts
    # Check if settled in target neighborhood
    is_safe = (np.abs(curr[0]) < 0.12) and (np.abs(curr[1]) < 0.3)
    return is_safe, traj, acts

# ------------------------------------------------------------------------------
# STEP 0: Compute Ground Truth Basin of Attraction (B*) via Dense Grid Search
# ------------------------------------------------------------------------------
print("="*70)
print("[Step 0/5] Computing Ground-Truth Viable Basin B* via Dense Grid Simulation...")
print("="*70)

GRID_RES = 60
theta_grid = np.linspace(-1.2, 1.2, GRID_RES)
omega_grid = np.linspace(-3.0, 3.0, GRID_RES)
ground_truth_basin = []

for th in theta_grid:
    for om in omega_grid:
        s_test = np.array([th, om])
        safe, _, _ = check_mpc_recovery(s_test, max_steps=30)
        if safe:
            ground_truth_basin.append(s_test)

B_star = np.array(ground_truth_basin)
TOTAL_VIABLE_CELLS = len(B_star)
print(f"  -> Ground Truth Basin B* identified: {TOTAL_VIABLE_CELLS} viable states ({TOTAL_VIABLE_CELLS / (GRID_RES**2) * 100:.1f}% of phase space).")

def estimate_basin_coverage(discovered_states):
    """Measures % of ground truth basin B* covered within epsilon-ball of discovered states"""
    if len(discovered_states) == 0:
        return 0.0
    disc_arr = np.array(discovered_states)
    covered = 0
    # Radius in normalized space
    disc_norm = disc_arr / np.array([1.2, 3.0])
    b_norm = B_star / np.array([1.2, 3.0])
    
    # Fast nearest neighbor check
    for pt in b_norm:
        dists = np.sum((disc_norm - pt)**2, axis=1)
        if np.min(dists) < 0.025: # threshold
            covered += 1
    return (covered / TOTAL_VIABLE_CELLS) * 100.0

# ------------------------------------------------------------------------------
# STEP 1: Running Sampling Protocols under Fixed Compute Budget (N_MPC = 1200)
# ------------------------------------------------------------------------------
MPC_BUDGET = 1200
print("\n" + "="*70)
print(f"[Step 1/5] Evaluating Sampling Protocols (Fixed Budget = {MPC_BUDGET} MPC Queries)...")
print("="*70)

# PROTOCOL A: Uniform Random Sampling
print("  Running Protocol A: Uniform Random Sampling...")
safe_set_A = []
queries_A = 0
while queries_A < MPC_BUDGET:
    th = np.random.uniform(-1.2, 1.2)
    om = np.random.uniform(-3.0, 3.0)
    s_cand = np.array([th, om])
    safe, _, _ = check_mpc_recovery(s_cand)
    queries_A += 1
    if safe:
        safe_set_A.append(s_cand)

cov_A = estimate_basin_coverage(safe_set_A)
yield_A = (len(safe_set_A) / MPC_BUDGET) * 100.0
print(f"  -> Group A finished: Discovered {len(safe_set_A)} states | Yield: {yield_A:.1f}% | Coverage of B*: {cov_A:.1f}%")

# PROTOCOL B: Reverse Dynamic Random Walk from Goal (Florensa style)
print("  Running Protocol B: Goal-Directed Reverse Random Walk...")
safe_set_B = [np.array([0.0, 0.0])]
pool_B = [np.array([0.0, 0.0])]
queries_B = 0

while queries_B < MPC_BUDGET:
    # Pick a known safe state
    idx = np.random.randint(len(pool_B))
    x_anchor = pool_B[idx]
    
    # Reverse dynamic step with random action
    u_rand = np.random.uniform(-u_max, u_max)
    x_prev = step_backward(x_anchor, u_rand)
    
    if abs(x_prev[0]) > 1.2 or abs(x_prev[1]) > 3.0:
        continue
        
    safe, _, _ = check_mpc_recovery(x_prev)
    queries_B += 1
    if safe:
        safe_set_B.append(x_prev)
        pool_B.append(x_prev)
        if len(pool_B) > 300: # maintain active buffer
            pool_B.pop(0)

cov_B = estimate_basin_coverage(safe_set_B)
yield_B = (len(safe_set_B) / MPC_BUDGET) * 100.0
print(f"  -> Group B finished: Discovered {len(safe_set_B)} states | Yield: {yield_B:.1f}% | Coverage of B*: {cov_B:.1f}%")

# PROTOCOL C: Backward Recovery Sampling with Frontier/Boundary Mining (BRS, Ours)
print("  Running Protocol C: Backward Recovery Sampling with Active Boundary Mining (BRS)...")
safe_set_C = [np.array([0.0, 0.0])]
frontier_C = [np.array([0.0, 0.0])]
failures_C = []
transitions_C = [] # records (state, action) for distillation
queries_C = 0

while queries_C < MPC_BUDGET:
    # Active Boundary Selection:
    # Sample from frontier, prioritizing states close to failure boundary if failures exist
    if len(failures_C) > 5 and np.random.rand() < 0.65:
        # Pick failure point, find closest frontier point
        f_idx = np.random.randint(len(failures_C))
        f_pt = failures_C[f_idx]
        dists = [np.linalg.norm(pt - f_pt) for pt in frontier_C]
        x_anchor = frontier_C[np.argmin(dists)]
    else:
        x_anchor = frontier_C[np.random.randint(len(frontier_C))]
        
    # Take reverse step under physical dynamic action
    u_rev = np.random.uniform(-u_max, u_max)
    x_prev = step_backward(x_anchor, u_rev)
    
    if abs(x_prev[0]) > 1.2 or abs(x_prev[1]) > 3.0:
        continue
        
    # Forward MPC verification (The rigorous viability certificate)
    safe, traj, acts = check_mpc_recovery(x_prev)
    queries_C += 1
    
    if safe:
        safe_set_C.append(x_prev)
        frontier_C.append(x_prev)
        for s_t, a_t in zip(traj[:-1], acts):
            transitions_C.append((s_t, a_t))
        if len(frontier_C) > 200:
            frontier_C.pop(0)
    else:
        failures_C.append(x_prev)
        if len(failures_C) > 200:
            failures_C.pop(0)

cov_C = estimate_basin_coverage(safe_set_C)
yield_C = (len(safe_set_C) / MPC_BUDGET) * 100.0
print(f"  -> Group C finished: Discovered {len(safe_set_C)} states | Yield: {yield_C:.1f}% | Coverage of B*: {cov_C:.1f}%")

# ------------------------------------------------------------------------------
# STEP 2: Distilling Group C Data into Policies (Group D: BC vs. Group E: Diffusion)
# ------------------------------------------------------------------------------
print("\n" + "="*70)
print("[Step 2/5] Training Distilled Policies on BRS Dataset (D_trans)...")
print("="*70)

D_states = np.array([t[0] for t in transitions_C], dtype=np.float32)
D_actions = np.array([t[1] for t in transitions_C], dtype=np.float32).reshape(-1, 1)
print(f"  -> Training Dataset Size: {len(D_states)} state-action transitions from Group C.")

# Group D: Deterministic BC (MLP / MSE)
class DeterministicBCNet(nn.Module):
    def __init__(self, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden), nn.Mish(),
            nn.Linear(hidden, hidden), nn.Mish(),
            nn.Linear(hidden, 1)
        )
    def forward(self, x): return self.net(x)

# Group E: Generative Flow Matching Recovery Policy (MGRD)
class RecoveryFlowNet(nn.Module):
    def __init__(self, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1 + 2 + 1, hidden), nn.Mish(),
            nn.Linear(hidden, hidden), nn.Mish(),
            nn.Linear(hidden, 1)
        )
    def forward(self, u_tau, x, tau):
        return self.net(torch.cat([u_tau, x, tau], dim=-1))

loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(torch.tensor(D_states), torch.tensor(D_actions)),
    batch_size=128, shuffle=True
)

m_bc = DeterministicBCNet()
opt_bc = torch.optim.Adam(m_bc.parameters(), lr=2e-3)

m_mgrd = RecoveryFlowNet()
opt_mgrd = torch.optim.Adam(m_mgrd.parameters(), lr=2e-3)

for epoch in range(15):
    for x_b, u_b in loader:
        # BC
        loss_bc = F.mse_loss(m_bc(x_b), u_b)
        opt_bc.zero_grad(); loss_bc.backward(); opt_bc.step()
        
        # CFM Flow Matching
        B = x_b.shape[0]
        tau = torch.rand(B, 1)
        u0 = torch.randn_like(u_b)
        u_tau = (1.0 - tau) * u0 + tau * u_b
        target_v = u_b - u0
        loss_cfm = F.mse_loss(m_mgrd(u_tau, x_b, tau), target_v)
        opt_mgrd.zero_grad(); loss_cfm.backward(); opt_mgrd.step()

print("  -> Both Group D (BC) and Group E (Recovery Diffusion) successfully distilled.")

def get_action_bc(state):
    with torch.no_grad():
        s_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        return float(np.clip(m_bc(s_t).numpy()[0, 0], -u_max, u_max))

def get_action_mgrd(state, steps=5):
    with torch.no_grad():
        s_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        u = torch.randn(1, 1)
        dt_sub = 1.0 / steps
        for i in range(steps):
            tau = torch.tensor([[i * dt_sub]], dtype=torch.float32)
            v = m_mgrd(u, s_t, tau)
            u = u + v * dt_sub
        return float(np.clip(u.numpy()[0, 0], -u_max, u_max))

# ------------------------------------------------------------------------------
# STEP 3: Closed-Loop Dynamic Stabilization across Discovered Basin
# ------------------------------------------------------------------------------
print("\n" + "="*70)
print("[Step 3/5] Evaluating Closed-Loop Dynamic Stabilization across 80 Test Perturbations...")
print("="*70)

N_EVAL = 80
eval_states = []
# Sample testing states uniformly across the ground truth basin B*
indices = np.random.choice(len(B_star), size=N_EVAL, replace=False)
for idx in indices:
    eval_states.append(B_star[idx])

def evaluate_policy_closed_loop(policy_fn, max_steps=45):
    succ = 0
    for s0 in eval_states:
        curr = s0.copy()
        safe = True
        for _ in range(max_steps):
            u = policy_fn(curr)
            curr = step_forward(curr, u)
            if abs(curr[0]) > np.pi * 0.8:
                safe = False
                break
        if safe and abs(curr[0]) < 0.15 and abs(curr[1]) < 0.35:
            succ += 1
    return (succ / N_EVAL) * 100.0

rate_mpc = evaluate_policy_closed_loop(solve_mpc)
rate_bc = evaluate_policy_closed_loop(get_action_bc)
rate_mgrd = evaluate_policy_closed_loop(get_action_mgrd)

print(f"  -> Online MPC Oracle (Upper bound):                  {rate_mpc:5.1f}%")
print(f"  -> Group D (Deterministic BC Policy):                {rate_bc:5.1f}%")
print(f"  -> Group E (MGRD Generative Recovery Flow):          {rate_mgrd:5.1f}%")

# ------------------------------------------------------------------------------
# STEP 4: Summary Table & Scientific Analysis
# ------------------------------------------------------------------------------
print("\n" + "="*70)
print("FINAL CONTROLLED EXPERIMENTAL RESULTS (GROUPS A - E):")
print("="*70)

summary_results = {
    "Group A (Uniform Random + MPC)": {
        "Sampling": "Uniform Phase Space",
        "Policy": "Online MPC",
        "MPC Queries": MPC_BUDGET,
        "Discovered Safe States": len(safe_set_A),
        "MPC Yield Rate (%)": yield_A,
        "Basin B* Coverage (%)": cov_A,
        "Closed-Loop Stability (%)": rate_mpc
    },
    "Group B (Reverse Random Walk + MPC)": {
        "Sampling": "Backward Dynamics Walk (Florensa)",
        "Policy": "Online MPC",
        "MPC Queries": MPC_BUDGET,
        "Discovered Safe States": len(safe_set_B),
        "MPC Yield Rate (%)": yield_B,
        "Basin B* Coverage (%)": cov_B,
        "Closed-Loop Stability (%)": rate_mpc
    },
    "Group C (BRS Active Boundary + MPC)": {
        "Sampling": "Backward Recovery Sampling (Ours)",
        "Policy": "Online MPC",
        "MPC Queries": MPC_BUDGET,
        "Discovered Safe States": len(safe_set_C),
        "MPC Yield Rate (%)": yield_C,
        "Basin B* Coverage (%)": cov_C,
        "Closed-Loop Stability (%)": rate_mpc
    },
    "Group D (BRS + Deterministic BC)": {
        "Sampling": "Backward Recovery Sampling",
        "Policy": "Distilled MLP (BC)",
        "MPC Queries": MPC_BUDGET,
        "Discovered Safe States": len(safe_set_C),
        "MPC Yield Rate (%)": yield_C,
        "Basin B* Coverage (%)": cov_C,
        "Closed-Loop Stability (%)": rate_bc
    },
    "Group E (BRS + Recovery Diffusion MGRD)": {
        "Sampling": "Backward Recovery Sampling",
        "Policy": "Generative Flow Matching",
        "MPC Queries": MPC_BUDGET,
        "Discovered Safe States": len(safe_set_C),
        "MPC Yield Rate (%)": yield_C,
        "Basin B* Coverage (%)": cov_C,
        "Closed-Loop Stability (%)": rate_mgrd
    }
}

print(json.dumps(summary_results, indent=2))
with open("results_brs_benchmark.json", "w") as f:
    json.dump(summary_results, f, indent=2)

# ------------------------------------------------------------------------------
# STEP 5: Visualizing Phase Space Attractors & Active Frontiers
# ------------------------------------------------------------------------------
print("\nGenerating Diagnostic Figure: brs_sampling_benchmark.png...")
fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

# Plot 1: Ground Truth Basin vs. Group A Uniform Sampling
ax = axes[0]
ax.scatter(B_star[:, 0], B_star[:, 1], c='#bdbdbd', s=12, alpha=0.3, label='Ground Truth Basin B*')
arr_A = np.array(safe_set_A)
ax.scatter(arr_A[:, 0], arr_A[:, 1], c='#e41a1c', s=15, alpha=0.6, label=f'Group A: Uniform ({cov_A:.1f}% cov)')
ax.set_title("Group A: Uniform Random Sampling\n(Low sample efficiency, sparse coverage)", fontsize=11, fontweight='bold')
ax.set_xlabel(r"$\theta$ (rad)"); ax.set_ylabel(r"$\omega$ (rad/s)")
ax.set_xlim(-1.2, 1.2); ax.set_ylim(-3.0, 3.0)
ax.legend(loc='upper right', fontsize=9)
ax.grid(True, linestyle=':', alpha=0.6)

# Plot 2: Group B (Goal Walk) vs. Group C (BRS Active Boundary)
ax = axes[1]
ax.scatter(B_star[:, 0], B_star[:, 1], c='#bdbdbd', s=12, alpha=0.3, label='Ground Truth Basin B*')
arr_B = np.array(safe_set_B)
arr_C = np.array(safe_set_C)
arr_fail_C = np.array(failures_C) if len(failures_C) > 0 else np.empty((0, 2))

ax.scatter(arr_B[:, 0], arr_B[:, 1], c='#377eb8', s=15, alpha=0.4, label=f'Group B: Goal Walk ({cov_B:.1f}% cov)')
ax.scatter(arr_C[:, 0], arr_C[:, 1], c='#4daf4a', s=18, alpha=0.8, label=f'Group C: BRS Active ({cov_C:.1f}% cov)')
if len(arr_fail_C) > 0:
    ax.scatter(arr_fail_C[:, 0], arr_fail_C[:, 1], c='#984ea3', marker='x', s=20, alpha=0.5, label='Frontier Failures (Boundary)')
ax.set_title("Group B vs. Group C (BRS Frontier)\n(Targeted expansion along viability boundary)", fontsize=11, fontweight='bold')
ax.set_xlabel(r"$\theta$ (rad)"); ax.set_ylabel(r"$\omega$ (rad/s)")
ax.set_xlim(-1.2, 1.2); ax.set_ylim(-3.0, 3.0)
ax.legend(loc='upper right', fontsize=9)
ax.grid(True, linestyle=':', alpha=0.6)

# Plot 3: Closed-Loop Stabilization Comparison
ax = axes[2]
methods = ['Group A\n(Uniform)', 'Group B\n(Goal Walk)', 'Group C\n(BRS+MPC)', 'Group D\n(BRS+BC)', 'Group E\n(BRS+MGRD)']
cov_bars = [cov_A, cov_B, cov_C, cov_C, cov_C]
stab_bars = [rate_mpc, rate_mpc, rate_mpc, rate_bc, rate_mgrd]
x_pos = np.arange(len(methods))
width = 0.35

ax.bar(x_pos - width/2, cov_bars, width, label='Basin B* Coverage (%)', color='#2b83ba', alpha=0.85)
ax.bar(x_pos + width/2, stab_bars, width, label='Closed-Loop Recovery (%)', color='#fdae61', alpha=0.85)

ax.set_title("Controlled Five-Way Benchmark Metrics", fontsize=11, fontweight='bold')
ax.set_xticks(x_pos)
ax.set_xticklabels(methods, fontsize=8.5)
ax.set_ylabel("Percentage (%)")
ax.set_ylim(0, 115)
ax.legend(loc='upper right', fontsize=9)
ax.grid(True, axis='y', linestyle=':', alpha=0.6)

for i in range(len(methods)):
    ax.text(x_pos[i] - width/2, cov_bars[i] + 2, f"{cov_bars[i]:.1f}%", ha='center', fontsize=8, fontweight='bold')
    ax.text(x_pos[i] + width/2, stab_bars[i] + 2, f"{stab_bars[i]:.1f}%", ha='center', fontsize=8, fontweight='bold')

plt.tight_layout()
plt.savefig("brs_sampling_benchmark.png", dpi=200)
print("  -> Saved diagnostic benchmark figure to brs_sampling_benchmark.png.")
print("\n" + "="*70)
print("BRS EXPERIMENTAL BENCHMARK COMPLETED SUCCESSFULLY!")
print("="*70)
