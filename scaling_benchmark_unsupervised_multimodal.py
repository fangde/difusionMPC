import time
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

# ==============================================================================
# SCALING BENCHMARK: UNSUPERVISED MULTIMODAL DISCOVERY (1,000 -> 10,000 MPC BUDGET)
#
# Comparing:
#   Method A: Uniform Random Sampling in Phase Space
#   Method B: Goal-Directed Reverse Brownian Walk (Florensa CoRL 2017)
#   Method C: Backward Recovery Sampling with Frontier Boundary Mining (BRS, Ours)
#
# Budgets: [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]
# ==============================================================================

np.random.seed(42)
torch.manual_seed(42)
device = torch.device('cpu')

m, damp, dt, u_max = 1.0, 0.5, 0.05, 8.0
obs_pos = torch.tensor([-0.6, 0.0], device=device)
obs_r = 0.28
H = 12

def check_collision_np(p):
    return np.linalg.norm(p[:2] - np.array([-0.6, 0.0])) < obs_r

def batch_autonomous_mpc(x0_tensor, iters=25):
    """
    Pure optimization-based MPC without any heuristic rules or mode injection.
    Solves non-convex obstacle avoidance trajectory optimization via Adam.
    """
    B = x0_tensor.shape[0]
    U = torch.zeros(B, H, 2, device=device, requires_grad=True)
    with torch.no_grad():
        U.data = torch.randn(B, H, 2, device=device) * 0.02
        
    opt = torch.optim.Adam([U], lr=0.45)
    
    for _ in range(iters):
        opt.zero_grad()
        U_clamped = torch.clamp(U, -u_max, u_max)
        x = x0_tensor.clone()
        cost = 0.0
        for k in range(H):
            u_k = U_clamped[:, k]
            ax = (u_k[:, 0] - damp * x[:, 2]) / m
            ay = (u_k[:, 1] - damp * x[:, 3]) / m
            x = x + dt * torch.stack([x[:, 2], x[:, 3], ax, ay], dim=-1)
            cost = cost + 3.0 * (x[:, 0]**2 + x[:, 1]**2) + 0.5 * (x[:, 2]**2 + x[:, 3]**2) + 0.05 * torch.sum(u_k**2, dim=-1)
            d_obs = torch.norm(x[:, :2] - obs_pos, dim=-1)
            viol = torch.relu(obs_r + 0.06 - d_obs)
            cost = cost + 350.0 * (viol**2)
        cost = cost + 12.0 * (x[:, 0]**2 + x[:, 1]**2)
        loss = cost.sum()
        loss.backward()
        opt.step()
        
    u0 = torch.clamp(U[:, 0], -u_max, u_max).detach()
    
    # Forward simulation check: does executing this policy safely avoid collision?
    x_next = x0_tensor.clone()
    ax_step = (u0[:, 0] - damp * x_next[:, 2]) / m
    ay_step = (u0[:, 1] - damp * x_next[:, 3]) / m
    x_next = x_next + dt * torch.stack([x_next[:, 2], x_next[:, 3], ax_step, ay_step], dim=-1)
    
    d_next = torch.norm(x_next[:, :2] - obs_pos, dim=-1)
    is_safe = (d_next >= obs_r).cpu().numpy()
    
    return u0.cpu().numpy(), is_safe

def step_backward_walk_np(state, k_steps):
    s = state.copy()
    for _ in range(k_steps):
        u = np.random.uniform(-u_max, u_max, size=2)
        px, py, vx, vy = s
        ax = (u[0] - damp * vx) / m
        ay = (u[1] - damp * vy) / m
        s = s - dt * np.array([vx, vy, ax, ay])
    return s

def evaluate_dataset_metrics(states_list, actions_list):
    if len(states_list) == 0:
        return {"n_shadow": 0, "n_up": 0, "n_down": 0, "balance": 0.0, "coverage_pct": 0.0}
        
    S = np.array(states_list)
    A = np.array(actions_list)
    
    # Identify samples trapped behind the obstacle
    in_shadow = (S[:, 0] >= -1.45) & (S[:, 0] <= -0.65) & (np.abs(S[:, 1]) <= 0.35)
    shadow_acts = A[in_shadow]
    
    n_up = int(np.sum(shadow_acts[:, 1] > 0.4))
    n_down = int(np.sum(shadow_acts[:, 1] < -0.4))
    n_shadow = n_up + n_down
    
    if n_shadow > 0:
        balance = float(2.0 * min(n_up, n_down) / n_shadow)
    else:
        balance = 0.0
        
    grid_occ = np.zeros((20, 20), dtype=bool)
    gx = np.clip(((S[:, 0] - (-1.5)) / 2.0 * 20).astype(int), 0, 19)
    gy = np.clip(((S[:, 1] - (-0.8)) / 1.6 * 20).astype(int), 0, 19)
    grid_occ[gx, gy] = True
    cov_ratio = float(np.sum(grid_occ) / 400.0 * 100.0)
    
    return {
        "n_shadow": n_shadow,
        "n_up": n_up,
        "n_down": n_down,
        "balance": balance,
        "coverage_pct": cov_ratio
    }

budgets = list(range(1000, 10001, 1000))
print("="*70)
print(f"STARTING SCALING BENCHMARK: 1k -> 10k MPC Queries (10 increments)")
print("="*70)

data_A = {"states": [], "actions": []}
data_B = {"states": [], "actions": []}
data_C = {"states": [], "actions": []}

pool_B = [np.array([0.0, 0.0, 0.0, 0.0])]
frontier_C = [np.array([0.0, 0.0, 0.0, 0.0])]
failures_C = []

results_A = []
results_B = []
results_C = []

CHUNK_SIZE = 1000

for stage_idx, total_budget in enumerate(budgets):
    t_start_stage = time.time()
    print(f"\n>>> Scaling Stage {stage_idx+1}/10: Total Budget = {total_budget} MPC Solves...")
    
    # --------------------------------------------------------------------------
    # (A) Uniform Random Sampling
    # --------------------------------------------------------------------------
    px_u = np.random.uniform(-1.5, 0.5, size=CHUNK_SIZE)
    py_u = np.random.uniform(-0.8, 0.8, size=CHUNK_SIZE)
    vx_u = np.random.uniform(-0.8, 0.8, size=CHUNK_SIZE)
    vy_u = np.random.uniform(-0.8, 0.8, size=CHUNK_SIZE)
    batch_cand_A = np.stack([px_u, py_u, vx_u, vy_u], axis=-1)
    
    valid_mask_A = ~np.array([check_collision_np(s) for s in batch_cand_A])
    batch_cand_A = batch_cand_A[valid_mask_A]
    
    u_A, safe_A = batch_autonomous_mpc(torch.tensor(batch_cand_A, dtype=torch.float32))
    for s_i, u_i, is_s in zip(batch_cand_A, u_A, safe_A):
        if is_s:
            data_A["states"].append(s_i)
            data_A["actions"].append(u_i)
            
    m_A = evaluate_dataset_metrics(data_A["states"], data_A["actions"])
    m_A["total_safe"] = len(data_A["states"])
    m_A["yield_pct"] = len(data_A["states"]) / total_budget * 100.0
    results_A.append(m_A)
    
    # --------------------------------------------------------------------------
    # (B) Goal-Directed Reverse Brownian Walk (Florensa CoRL 2017: T_walk steps)
    # --------------------------------------------------------------------------
    cand_B = []
    for _ in range(CHUNK_SIZE):
        idx = np.random.randint(len(pool_B))
        anchor = pool_B[idx]
        k_walk = np.random.randint(8, 20)
        s_prev = step_backward_walk_np(anchor, k_walk)
        cand_B.append(s_prev)
    cand_B = np.array(cand_B)
    
    valid_mask_B = ~np.array([check_collision_np(s) for s in cand_B])
    cand_B_eval = cand_B[valid_mask_B]
    
    u_B, safe_B = batch_autonomous_mpc(torch.tensor(cand_B_eval, dtype=torch.float32))
    for s_i, u_i, is_s in zip(cand_B_eval, u_B, safe_B):
        if is_s:
            data_B["states"].append(s_i)
            data_B["actions"].append(u_i)
            pool_B.append(s_i)
            if len(pool_B) > 2000:
                pool_B.pop(0)
                
    m_B = evaluate_dataset_metrics(data_B["states"], data_B["actions"])
    m_B["total_safe"] = len(data_B["states"])
    m_B["yield_pct"] = len(data_B["states"]) / total_budget * 100.0
    results_B.append(m_B)
    
    # --------------------------------------------------------------------------
    # (C) Backward Recovery Sampling with Active Boundary Frontier (BRS)
    # --------------------------------------------------------------------------
    cand_C = []
    for _ in range(CHUNK_SIZE):
        if len(failures_C) > 10 and np.random.rand() < 0.65:
            f_pt = failures_C[np.random.randint(len(failures_C))]
            sub_idx = np.random.choice(len(frontier_C), size=min(30, len(frontier_C)), replace=False)
            sub_frontier = [frontier_C[i] for i in sub_idx]
            dists = [np.linalg.norm(pt[:2] - f_pt[:2]) for pt in sub_frontier]
            anchor = sub_frontier[np.argmin(dists)]
            k_walk = np.random.randint(4, 12) # fine boundary expansion
        else:
            anchor = frontier_C[np.random.randint(len(frontier_C))]
            k_walk = np.random.randint(8, 20)
            
        s_prev = step_backward_walk_np(anchor, k_walk)
        cand_C.append(s_prev)
    cand_C = np.array(cand_C)
    
    valid_mask_C = ~np.array([check_collision_np(s) for s in cand_C])
    cand_C_eval = cand_C[valid_mask_C]
    # Keep track of points that landed inside collision as failures
    for s_col in cand_C[~valid_mask_C]:
        failures_C.append(s_col)
        
    u_C, safe_C = batch_autonomous_mpc(torch.tensor(cand_C_eval, dtype=torch.float32))
    for s_i, u_i, is_s in zip(cand_C_eval, u_C, safe_C):
        if is_s:
            data_C["states"].append(s_i)
            data_C["actions"].append(u_i)
            frontier_C.append(s_i)
            if len(frontier_C) > 2000:
                frontier_C.pop(0)
        else:
            failures_C.append(s_i)
            if len(failures_C) > 2000:
                failures_C.pop(0)
                
    m_C = evaluate_dataset_metrics(data_C["states"], data_C["actions"])
    m_C["total_safe"] = len(data_C["states"])
    m_C["yield_pct"] = len(data_C["states"]) / total_budget * 100.0
    results_C.append(m_C)
    
    elapsed_s = time.time() - t_start_stage
    print(f"  [Uniform A] Safe: {m_A['total_safe']:5d} | Shadow: {m_A['n_shadow']:4d} (Up: {m_A['n_up']:3d}, Down: {m_A['n_down']:3d}) | Balance: {m_A['balance']:.2f} | Cov: {m_A['coverage_pct']:.1f}%")
    print(f"  [GoalWalk B] Safe: {m_B['total_safe']:5d} | Shadow: {m_B['n_shadow']:4d} (Up: {m_B['n_up']:3d}, Down: {m_B['n_down']:3d}) | Balance: {m_B['balance']:.2f} | Cov: {m_B['coverage_pct']:.1f}%")
    print(f"  [BRS Ours C] Safe: {m_C['total_safe']:5d} | Shadow: {m_C['n_shadow']:4d} (Up: {m_C['n_up']:3d}, Down: {m_C['n_down']:3d}) | Balance: {m_C['balance']:.2f} | Cov: {m_C['coverage_pct']:.1f}%")
    print(f"  Stage completed in {elapsed_s:.1f}s.")

scaling_data = {
    "budgets": budgets,
    "method_A": results_A,
    "method_B": results_B,
    "method_C": results_C
}
with open("results_scaling_unsupervised_multimodal.json", "w") as f:
    json.dump(scaling_data, f, indent=2)

# ------------------------------------------------------------------------------
# 4. Train Diffusion on 10k Datasets
# ------------------------------------------------------------------------------
print("\n" + "="*70)
print("[Policy Distillation] Training Diffusion Policies on 10k Datasets...")
print("="*70)

class FlowMatchingPolicy(nn.Module):
    def __init__(self, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 + 4 + 1, hidden), nn.Mish(),
            nn.Linear(hidden, hidden), nn.Mish(),
            nn.Linear(hidden, 2)
        )
    def forward(self, u_tau, x, tau):
        return self.net(torch.cat([u_tau, x, tau], dim=-1))

def train_cfm_policy(dataset_dict, epochs=14):
    X = torch.tensor(np.array(dataset_dict["states"]), dtype=torch.float32)
    U = torch.tensor(np.array(dataset_dict["actions"]), dtype=torch.float32)
    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X, U), batch_size=128, shuffle=True)
    model = FlowMatchingPolicy().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    
    for epoch in range(epochs):
        for x_b, u_b in loader:
            B = x_b.shape[0]
            tau = torch.rand(B, 1)
            u0 = torch.randn_like(u_b)
            u_tau = (1.0 - tau) * u0 + tau * u_b
            target_v = u_b - u0
            pred_v = model(u_tau, x_b, tau)
            loss = F.mse_loss(pred_v, target_v)
            opt.zero_grad(); loss.backward(); opt.step()
    return model

print("  Training Diffusion Policy on Group A (Uniform 10k)...")
model_A = train_cfm_policy(data_A)

print("  Training Diffusion Policy on Group B (Goal-Walk 10k)...")
model_B = train_cfm_policy(data_B)

print("  Training Diffusion Policy on Group C (BRS 10k)...")
model_C = train_cfm_policy(data_C)

# Sample from trained Diffusion models at x_test = [-1.1, 0, 0, 0]
x_test = np.array([-1.1, 0.0, 0.0, 0.0])

def sample_diffusion_actions(model, x_query, n_samples=300, num_steps=10):
    with torch.no_grad():
        x_t = torch.tensor(x_query, dtype=torch.float32).unsqueeze(0).repeat(n_samples, 1)
        u = torch.randn(n_samples, 2)
        dt_sub = 1.0 / num_steps
        for i in range(num_steps):
            tau = torch.full((n_samples, 1), i * dt_sub, dtype=torch.float32)
            v = model(u, x_t, tau)
            u = u + v * dt_sub
        return np.clip(u.numpy(), -u_max, u_max)

actions_gen_A = sample_diffusion_actions(model_A, x_test)
actions_gen_B = sample_diffusion_actions(model_B, x_test)
actions_gen_C = sample_diffusion_actions(model_C, x_test)

# ------------------------------------------------------------------------------
# 5. Diagnostic Visualization
# ------------------------------------------------------------------------------
print("\nGenerating Diagnostic Figures: scaling_multimodal_discovery.png...")

fig, axes = plt.subplots(2, 2, figsize=(15, 12))

b_list = budgets
bal_A = [r["balance"] for r in results_A]
bal_B = [r["balance"] for r in results_B]
bal_C = [r["balance"] for r in results_C]

ax = axes[0, 0]
ax.plot(b_list, bal_A, 'r-o', linewidth=2.2, label='Method A: Uniform Random')
ax.plot(b_list, bal_B, 'b-s', linewidth=2.2, label='Method B: Goal-Walk (Florensa)')
ax.plot(b_list, bal_C, 'g-^', linewidth=2.5, label='Method C: BRS Frontier (Ours)')
ax.axhline(1.0, color='gray', linestyle='--', alpha=0.5, label='Perfect 50/50 Balance')
ax.set_title("1. Unsupervised Multimodal Balance vs. Budget", fontsize=11, fontweight='bold')
ax.set_xlabel("Cumulative MPC Solves"); ax.set_ylabel("Multimodal Balance [0, 1]")
ax.set_ylim(-0.05, 1.1)
ax.legend(loc='lower right', fontsize=9)
ax.grid(True, linestyle=':', alpha=0.6)

ax = axes[0, 1]
n_shadow_A = [r["n_shadow"] for r in results_A]
n_shadow_B = [r["n_shadow"] for r in results_B]
n_shadow_C = [r["n_shadow"] for r in results_C]

ax.plot(b_list, n_shadow_A, 'r-o', linewidth=2.0, label=f'Method A (Final: {n_shadow_A[-1]})')
ax.plot(b_list, n_shadow_B, 'b-s', linewidth=2.0, label=f'Method B (Final: {n_shadow_B[-1]})')
ax.plot(b_list, n_shadow_C, 'g-^', linewidth=2.5, label=f'Method C (Final: {n_shadow_C[-1]})')
ax.set_title("2. Bifurcated Recovery Samples Behind Obstacle", fontsize=11, fontweight='bold')
ax.set_xlabel("Cumulative MPC Solves"); ax.set_ylabel("Number of Discovered Samples")
ax.legend(loc='upper left', fontsize=9)
ax.grid(True, linestyle=':', alpha=0.6)

ax = axes[1, 0]
cov_A = [r["coverage_pct"] for r in results_A]
cov_B = [r["coverage_pct"] for r in results_B]
cov_C = [r["coverage_pct"] for r in results_C]

ax.plot(b_list, cov_A, 'r-o', linewidth=2.0, label=f'Method A: Uniform ({cov_A[-1]:.1f}%)')
ax.plot(b_list, cov_B, 'b-s', linewidth=2.0, label=f'Method B: Goal-Walk ({cov_B[-1]:.1f}%)')
ax.plot(b_list, cov_C, 'g-^', linewidth=2.5, label=f'Method C: BRS Active ({cov_C[-1]:.1f}%)')
ax.set_title("3. Basin Spatial Coverage vs. Budget", fontsize=11, fontweight='bold')
ax.set_xlabel("Cumulative MPC Solves"); ax.set_ylabel("Spatial Coverage (%)")
ax.legend(loc='lower right', fontsize=9)
ax.grid(True, linestyle=':', alpha=0.6)

ax = axes[1, 1]
ax.scatter(actions_gen_A[:, 0], actions_gen_A[:, 1], color='#e41a1c', alpha=0.35, label=f'Diffusion on A (Bal={bal_A[-1]:.2f})')
ax.scatter(actions_gen_B[:, 0], actions_gen_B[:, 1], color='#377eb8', alpha=0.45, label=f'Diffusion on B (Bal={bal_B[-1]:.2f})')
ax.scatter(actions_gen_C[:, 0], actions_gen_C[:, 1], color='#4daf4a', alpha=0.65, label=f'Diffusion on C (Bal={bal_C[-1]:.2f})')
ax.axhline(0, color='black', linestyle='--', alpha=0.4)
ax.set_title("4. Sampled Actions from Distilled Diffusion at x_test = [-1.1, 0]", fontsize=11, fontweight='bold')
ax.set_xlabel("Action u_x (N)"); ax.set_ylabel("Action u_y (N)")
ax.legend(loc='upper right', fontsize=8.5)
ax.grid(True, linestyle=':', alpha=0.6)

plt.tight_layout()
plt.savefig("scaling_multimodal_discovery.png", dpi=200)
print("  -> Saved figure to scaling_multimodal_discovery.png.")
print("\n" + "="*70)
print("SCALING EXPERIMENT FINISHED SUCCESSFULLY!")
print("="*70)
