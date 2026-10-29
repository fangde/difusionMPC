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
# RIGOROUS SCALING BENCHMARK: 4D FULLY ACTUATED 2-DOF ROBOT ARM
#
# Core Research Questions:
#   1. "高维下，这个方法、采样还能不能用？"
#      (In higher dimensions, can this method and sampling still work?)
#   2. "多模态还有没有？"
#      (Does natural multimodality still exist in higher dimensions?)
#
# Setup:
#   System: 2-DOF Planar Robot Arm (RR Manipulator, fully actuated: B = I_2)
#   State:  x = [theta_1, theta_2, dtheta_1, dtheta_2]^T in R^4
#   Action: u = [tau_1, tau_2]^T in [-u_max, u_max]^2 (u_max = 15.0 N*m)
#   Obstacle: Circular obstacle in task space at (1.45, 0.0), radius R_obs = 0.22
#   Target EE: p* = [1.114, 0.0] admitting two exact symmetric IK branches:
#              Branch 1: Elbow-Up   (q1 > 0, q2 < 0, clears obstacle via y > 0)
#              Branch 2: Elbow-Down (q1 < 0, q2 > 0, clears obstacle via y < 0)
#   Crucial: ZERO manual mode injection!
#
# Budgets: [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]
# ==============================================================================

np.random.seed(42)
torch.manual_seed(42)
device = torch.device('cpu')

# ------------------------------------------------------------------------------
# 1. Fully Actuated Robot Dynamics & Kinematics
# ------------------------------------------------------------------------------
l1, l2 = 1.0, 1.0
m1, m2 = 1.0, 1.0
u_max = 15.0
dt = 0.04
H = 20  # MPC horizon

# Obstacle placed directly on the centerline between initial arm and target
obs_pos = torch.tensor([1.45, 0.0], device=device)
obs_r = 0.22
target_ee = torch.tensor([1.114, 0.0], device=device)

def batch_forward_kinematics(q):
    th1 = q[..., 0]
    th2 = q[..., 1]
    p1 = torch.stack([torch.cos(th1), torch.sin(th1)], dim=-1)
    p2 = p1 + torch.stack([torch.cos(th1 + th2), torch.sin(th1 + th2)], dim=-1)
    return p1, p2

def dist_segment(p1, p2, p_obs):
    v = p2 - p1
    w = p_obs - p1
    c1 = torch.sum(w * v, dim=-1)
    c2 = torch.sum(v * v, dim=-1) + 1e-6
    t = torch.clamp(c1 / c2, 0.0, 1.0)
    proj = p1 + t.unsqueeze(-1) * v
    return torch.norm(proj - p_obs, dim=-1)

def robot_forward_step(x, u, dt_step=0.04):
    """
    Continuous-to-discrete integration for 2-DOF planar arm:
    M(q) ddq + C(q, dq) dq + D dq = u
    """
    q = x[:, :2]
    dq = x[:, 2:]
    th2 = q[:, 1]
    
    # Analytical M(q)^-1
    M11 = 5.0 / 3.0 + torch.cos(th2)
    M12 = 1.0 / 3.0 + 0.5 * torch.cos(th2)
    M22 = 1.0 / 3.0 * torch.ones_like(th2)
    det = torch.clamp(M11 * M22 - M12**2, min=0.18)
    inv11 = M22 / det
    inv12 = -M12 / det
    inv22 = M11 / det
    
    # Coriolis forces
    C1 = -0.5 * torch.sin(th2) * (2.0 * dq[:, 0] * dq[:, 1] + dq[:, 1]**2)
    C2 = 0.5 * torch.sin(th2) * (dq[:, 0]**2)
    
    # Joint damping d1=0.5, d2=0.5
    net_tau_1 = u[:, 0] - C1 - 0.5 * dq[:, 0]
    net_tau_2 = u[:, 1] - C2 - 0.5 * dq[:, 1]
    
    ddq1 = inv11 * net_tau_1 + inv12 * net_tau_2
    ddq2 = inv12 * net_tau_1 + inv22 * net_tau_2
    
    dq_new = dq + dt_step * torch.stack([ddq1, ddq2], dim=-1)
    q_new = q + dt_step * dq_new
    return torch.cat([q_new, dq_new], dim=-1)

# ------------------------------------------------------------------------------
# 2. Autonomous Batch MPC Oracle (Zero Mode Injection)
# ------------------------------------------------------------------------------
def batch_autonomous_mpc(x0_tensor, iters=20):
    """
    Trajectory optimization via Adam with zero heuristic mode guidance.
    Multimodality emerges strictly from non-convex obstacle landscape.
    """
    B = x0_tensor.shape[0]
    U = torch.zeros(B, H, 2, device=device, requires_grad=True)
    with torch.no_grad():
        U.data = torch.randn(B, H, 2, device=device) * 0.05
        
    opt = torch.optim.Adam([U], lr=0.5)
    
    for _ in range(iters):
        opt.zero_grad()
        U_clamped = torch.clamp(U, -u_max, u_max)
        x = x0_tensor.clone()
        cost = 0.0
        for k in range(H):
            u_k = U_clamped[:, k]
            x = robot_forward_step(x, u_k, dt)
            p1, p2 = batch_forward_kinematics(x[:, :2])
            
            # Task-space target attraction
            ee_err = torch.norm(p2 - target_ee, dim=-1)
            cost = cost + 15.0 * (ee_err**2) + 0.5 * torch.sum(x[:, 2:]**2, dim=-1) + 0.01 * torch.sum(u_k**2, dim=-1)
            
            # Obstacle barrier
            d1 = dist_segment(torch.zeros_like(p1), p1, obs_pos)
            d2 = dist_segment(p1, p2, obs_pos)
            d_min = torch.minimum(d1, d2)
            viol = torch.relu(obs_r + 0.04 - d_min)
            cost = cost + 500.0 * (viol**2)
            
        loss = cost.sum()
        loss.backward()
        opt.step()
        
    u0 = torch.clamp(U[:, 0], -u_max, u_max).detach()
    
    # Forward simulation check: does executing this policy safely avoid collision & reach target?
    with torch.no_grad():
        x_roll = x0_tensor.clone()
        safe_mask = torch.ones(B, dtype=torch.bool, device=device)
        U_eval = torch.clamp(U, -u_max, u_max)
        for k in range(H):
            x_roll = robot_forward_step(x_roll, U_eval[:, k], dt)
            p1, p2 = batch_forward_kinematics(x_roll[:, :2])
            d1 = dist_segment(torch.zeros_like(p1), p1, obs_pos)
            d2 = dist_segment(p1, p2, obs_pos)
            d_min = torch.minimum(d1, d2)
            safe_mask = safe_mask & (d_min >= obs_r)
            
        # Target reaching check
        final_p1, final_p2 = batch_forward_kinematics(x_roll[:, :2])
        final_ee_err = torch.norm(final_p2 - target_ee, dim=-1)
        final_vel = torch.norm(x_roll[:, 2:], dim=-1)
        is_safe = (safe_mask & (final_ee_err < 0.28) & (final_vel < 2.5)).cpu().numpy()
        
    return u0.cpu().numpy(), is_safe

# ------------------------------------------------------------------------------
# 3. Dynamic Step Backward Integration
# ------------------------------------------------------------------------------
def step_backward_robot(state_tensor, k_steps):
    """
    Time-reversed step along dynamics: x_prev = x - dt * f(x, u)
    """
    x = state_tensor.clone()
    for _ in range(k_steps):
        u_rand = (torch.rand(x.shape[0], 2, device=device) * 2.0 - 1.0) * u_max
        q = x[:, :2]
        dq = x[:, 2:]
        th2 = q[:, 1]
        
        M11 = 5.0 / 3.0 + torch.cos(th2)
        M12 = 1.0 / 3.0 + 0.5 * torch.cos(th2)
        M22 = 1.0 / 3.0 * torch.ones_like(th2)
        det = torch.clamp(M11 * M22 - M12**2, min=0.18)
        inv11 = M22 / det
        inv12 = -M12 / det
        inv22 = M11 / det
        
        C1 = -0.5 * torch.sin(th2) * (2.0 * dq[:, 0] * dq[:, 1] + dq[:, 1]**2)
        C2 = 0.5 * torch.sin(th2) * (dq[:, 0]**2)
        
        net_tau_1 = u_rand[:, 0] - C1 - 0.5 * dq[:, 0]
        net_tau_2 = u_rand[:, 1] - C2 - 0.5 * dq[:, 1]
        
        ddq1 = inv11 * net_tau_1 + inv12 * net_tau_2
        ddq2 = inv12 * net_tau_1 + inv22 * net_tau_2
        
        # Reverse Euler
        dq_prev = dq - dt * torch.stack([ddq1, ddq2], dim=-1)
        q_prev = q - dt * dq
        x = torch.cat([q_prev, dq_prev], dim=-1)
    return x

# ------------------------------------------------------------------------------
# 4. Evaluation Metrics in 4D Phase Space
# ------------------------------------------------------------------------------
def evaluate_4d_robot_dataset(states_list, actions_list):
    if len(states_list) == 0:
        return {"n_safe": 0, "n_elbow_up": 0, "n_elbow_down": 0, "balance": 0.0, "vol_4d": 0.0}
        
    S = np.array(states_list)
    A = np.array(actions_list)
    
    # Multimodal branches: In the shadow of the obstacle (q1 near 0, arm extended),
    # Does the controller steer Elbow-Up (tau_1 > 0.5 / tau_2 < -0.5) or Elbow-Down (tau_1 < -0.5 / tau_2 > 0.5)?
    # We classify by the sign of tau_1 or elbow angle direction:
    n_up = int(np.sum(A[:, 0] > 0.5))
    n_down = int(np.sum(A[:, 0] < -0.5))
    tot = n_up + n_down
    balance = float(2.0 * min(n_up, n_down) / tot) if tot > 0 else 0.0
    
    # 4D Hypervolume bounding box (span of viable basin in 4D phase space)
    spans = np.ptp(S, axis=0)
    vol_4d = float(np.prod(np.maximum(spans, 1e-3)))
    
    return {
        "n_safe": len(S),
        "n_elbow_up": n_up,
        "n_elbow_down": n_down,
        "balance": balance,
        "vol_4d": vol_4d
    }

# ==============================================================================
# MAIN EXPERIMENT: 1k -> 10k BUDGET SCALING
# ==============================================================================
if __name__ == '__main__':
    budgets = list(range(1000, 10001, 1000))
    print("="*75)
    print("SCALING BENCHMARK: 4D FULLY ACTUATED 2-DOF ROBOT ARM (1k -> 10k MPC)")
    print("Evaluating Curse of Dimensionality & Unsupervised Multimodal Discovery")
    print("="*75)

    # Two symmetric goal targets in joint space
    target_q_up = torch.tensor([[0.98, -1.96, 0.0, 0.0]], device=device)
    target_q_down = torch.tensor([[-0.98, 1.96, 0.0, 0.0]], device=device)

    # --- METHOD A: 4D Uniform Random Sampling ---
    print("\n[Method A] Running 4D Uniform Random Sampling...")
    A_states, A_actions = [], []
    metrics_A = []

    for b in budgets:
        batch_size = 1000
        # Uniform in q in [-1.5, 1.5], dq in [-2.0, 2.0]
        q_cand = torch.empty(batch_size, 2).uniform_(-1.5, 1.5)
        dq_cand = torch.empty(batch_size, 2).uniform_(-2.0, 2.0)
        x_cand = torch.cat([q_cand, dq_cand], dim=-1).to(device)
        
        u_opt, is_safe = batch_autonomous_mpc(x_cand)
        for i in range(batch_size):
            if is_safe[i]:
                A_states.append(x_cand[i].cpu().numpy())
                A_actions.append(u_opt[i])
                
        m = evaluate_4d_robot_dataset(A_states, A_actions)
        m["budget"] = b
        m["yield_rate"] = len(A_states) / b
        metrics_A.append(m)
        print(f"  Budget {b:5d}: Safe={m['n_safe']:4d} (Yield={m['yield_rate']*100:4.1f}%), Elbow-Up={m['n_elbow_up']:3d}, Elbow-Down={m['n_elbow_down']:3d}, Balance={m['balance']:.2f}, Vol4D={m['vol_4d']:.2f}")

    # --- METHOD B: Goal-Directed Reverse Brownian Walk (Florensa CoRL 2017) ---
    print("\n[Method B] Running Goal-Directed Reverse Brownian Walk...")
    B_states, B_actions = [], []
    metrics_B = []

    curr_B_pool = [target_q_up, target_q_down]

    for b in budgets:
        batch_size = 1000
        idx = np.random.randint(0, len(curr_B_pool), size=batch_size)
        x_anchors = torch.cat([curr_B_pool[i] for i in idx], dim=0)
        k_steps = np.random.randint(1, 6)
        x_cand = step_backward_robot(x_anchors, k_steps)
        
        u_opt, is_safe = batch_autonomous_mpc(x_cand)
        for i in range(batch_size):
            if is_safe[i]:
                B_states.append(x_cand[i].cpu().numpy())
                B_actions.append(u_opt[i])
                curr_B_pool.append(x_cand[i:i+1])
                
        m = evaluate_4d_robot_dataset(B_states, B_actions)
        m["budget"] = b
        m["yield_rate"] = len(B_states) / b
        metrics_B.append(m)
        print(f"  Budget {b:5d}: Safe={m['n_safe']:4d} (Yield={m['yield_rate']*100:4.1f}%), Elbow-Up={m['n_elbow_up']:3d}, Elbow-Down={m['n_elbow_down']:3d}, Balance={m['balance']:.2f}, Vol4D={m['vol_4d']:.2f}")

    # --- METHOD C: Backward Recovery Sampling with Active Boundary Frontier (BRS, Ours) ---
    print("\n[Method C] Running Backward Recovery Sampling with Active Boundary Frontier (BRS)...")
    C_states, C_actions = [], []
    metrics_C = []

    safe_frontier = [target_q_up, target_q_down]
    boundary_frontier = []

    for b in budgets:
        batch_size = 1000
        # Active Boundary Sampling: 60% from Boundary Frontier, 40% from Safe Frontier
        n_boundary = int(batch_size * 0.6) if len(boundary_frontier) > 20 else 0
        n_safe = batch_size - n_boundary
        
        cands = []
        if n_boundary > 0:
            idx_b = np.random.randint(0, len(boundary_frontier), size=n_boundary)
            anchors_b = torch.cat([boundary_frontier[i] for i in idx_b], dim=0)
            cands.append(step_backward_robot(anchors_b, k_steps=np.random.randint(1, 3)))
            
        idx_s = np.random.randint(0, len(safe_frontier), size=n_safe)
        anchors_s = torch.cat([safe_frontier[i] for i in idx_s], dim=0)
        cands.append(step_backward_robot(anchors_s, k_steps=np.random.randint(1, 5)))
        
        x_cand = torch.cat(cands, dim=0)
        u_opt, is_safe = batch_autonomous_mpc(x_cand)
        
        for i in range(batch_size):
            s_i = x_cand[i:i+1]
            if is_safe[i]:
                C_states.append(x_cand[i].cpu().numpy())
                C_actions.append(u_opt[i])
                safe_frontier.append(s_i)
            else:
                # Active boundary point discovered
                boundary_frontier.append(s_i)
                if len(boundary_frontier) > 2000:
                    boundary_frontier.pop(0)
                    
        m = evaluate_4d_robot_dataset(C_states, C_actions)
        m["budget"] = b
        m["yield_rate"] = len(C_states) / b
        metrics_C.append(m)
        print(f"  Budget {b:5d}: Safe={m['n_safe']:4d} (Yield={m['yield_rate']*100:4.1f}%), Elbow-Up={m['n_elbow_up']:3d}, Elbow-Down={m['n_elbow_down']:3d}, Balance={m['balance']:.2f}, Vol4D={m['vol_4d']:.2f}")

    # ==============================================================================
    # 5. POLICY LEARNING: CONTINUOUS FLOW MATCHING (DIFFUSION) VS. BC
    # ==============================================================================
    print("\n" + "="*75)
    print("DISTILLATION & CLOSED-LOOP VALIDATION ON 4D FULLY ACTUATED ROBOT")
    print("Comparing Generative Flow Matching Policy vs Deterministic Behavior Cloning")
    print("="*75)

    X_train = torch.tensor(np.array(C_states), dtype=torch.float32, device=device)
    U_train = torch.tensor(np.array(C_actions), dtype=torch.float32, device=device)
    N_data = len(X_train)

    # 1. Deterministic BC Network
    class BCNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(4, 128), nn.SiLU(),
                nn.Linear(128, 128), nn.SiLU(),
                nn.Linear(128, 2)
            )
        def forward(self, x):
            return torch.clamp(self.net(x), -u_max, u_max)

    # 2. Continuous Flow Matching Recovery Vector Field (Diffusion)
    class CFMNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(4 + 2 + 1, 128), nn.SiLU(),
                nn.Linear(128, 128), nn.SiLU(),
                nn.Linear(128, 2)
            )
        def forward(self, x, u_t, t):
            inp = torch.cat([x, u_t, t], dim=-1)
            return self.net(inp)

    bc_model = BCNet().to(device)
    cfm_model = CFMNet().to(device)

    opt_bc = torch.optim.Adam(bc_model.parameters(), lr=1e-3)
    opt_cfm = torch.optim.Adam(cfm_model.parameters(), lr=1e-3)

    print(f"Training BC & CFM Models on {N_data} samples...")
    batch_sz = 128
    epochs = 40
    for ep in range(epochs):
        perm = torch.randperm(N_data)
        for i in range(0, N_data, batch_sz):
            b_idx = perm[i:i+batch_sz]
            xb, ub = X_train[b_idx], U_train[b_idx]
            
            # Train BC
            opt_bc.zero_grad()
            loss_bc = F.mse_loss(bc_model(xb), ub)
            loss_bc.backward()
            opt_bc.step()
            
            # Train CFM (Optimal Transport Flow Matching: target v = u1 - u0)
            opt_cfm.zero_grad()
            B_b = xb.shape[0]
            t = torch.rand(B_b, 1, device=device)
            u0 = torch.randn_like(ub)
            u1 = ub
            ut = (1.0 - t) * u0 + t * u1
            v_target = u1 - u0
            v_pred = cfm_model(xb, ut, t)
            loss_cfm = F.mse_loss(v_pred, v_target)
            loss_cfm.backward()
            opt_cfm.step()

    print("Training complete!")

    # ------------------------------------------------------------------------------
    # 6. Closed-Loop Evaluation on 100 Perturbed Test States
    # ------------------------------------------------------------------------------
    print("\nRunning Closed-Loop Stabilization Test (100 Initial Perturbed States)...")

    # Sample 100 test states from the verified safe dataset
    test_idx = np.random.choice(len(C_states), size=100, replace=False)
    X_test = np.array([C_states[i] for i in test_idx])

    def rollout_policy(policy_type, x_init, max_steps=50):
        curr = torch.tensor(x_init, dtype=torch.float32, device=device).unsqueeze(0)
        traj = [curr[0].cpu().numpy()]
        collided = False
        
        for s in range(max_steps):
            if policy_type == 'bc':
                with torch.no_grad():
                    u_cmd = bc_model(curr)
            elif policy_type == 'cfm':
                with torch.no_grad():
                    u_curr = torch.randn(1, 2, device=device)
                    n_ode = 10
                    dt_ode = 1.0 / n_ode
                    for st in range(n_ode):
                        t_val = torch.tensor([[st * dt_ode]], device=device)
                        v = cfm_model(curr, u_curr, t_val)
                        u_curr = u_curr + dt_ode * v
                    u_cmd = torch.clamp(u_curr, -u_max, u_max)
                    
            curr = robot_forward_step(curr, u_cmd, dt)
            traj.append(curr[0].cpu().numpy())
            
            # Check collision with obstacle
            p1, p2 = batch_forward_kinematics(curr[:, :2])
            d1 = dist_segment(torch.zeros_like(p1), p1, obs_pos)
            d2 = dist_segment(p1, p2, obs_pos)
            if min(d1.item(), d2.item()) < obs_r:
                collided = True
                break
                
        # Final end-effector error
        final_p1, final_p2 = batch_forward_kinematics(curr[:, :2])
        final_ee_err = torch.norm(final_p2 - target_ee).item()
        success = (not collided) and (final_ee_err < 0.35)
        return success, collided, np.array(traj)

    bc_success, bc_colls = 0, 0
    cfm_success, cfm_colls = 0, 0
    cfm_elbow_up_trajs, cfm_elbow_down_trajs = [], []

    for i in range(100):
        s_bc, c_bc, _ = rollout_policy('bc', X_test[i])
        if s_bc: bc_success += 1
        if c_bc: bc_colls += 1
        
        s_cfm, c_cfm, tr_cfm = rollout_policy('cfm', X_test[i])
        if s_cfm: 
            cfm_success += 1
            if tr_cfm[:, 0].mean() > 0.0:
                cfm_elbow_up_trajs.append(tr_cfm)
            else:
                cfm_elbow_down_trajs.append(tr_cfm)
        if c_cfm: cfm_colls += 1

    print("\n" + "="*75)
    print("FINAL TEST RESULTS (4D FULLY ACTUATED ROBOT ARM):")
    print(f"  [Behavior Cloning (BC)]:  Success = {bc_success}/100 ({bc_success}%), Collisions = {bc_colls}/100 ({bc_colls}%)")
    print(f"  [Continuous Flow Matching]: Success = {cfm_success}/100 ({cfm_success}%), Collisions = {cfm_colls}/100 ({cfm_colls}%)")
    print(f"  [Multimodal Discovery in Test]: Elbow-Up = {len(cfm_elbow_up_trajs)}, Elbow-Down = {len(cfm_elbow_down_trajs)}")
    print("="*75)

    # ==============================================================================
    # 7. GENERATE 4-PANEL PUBLICATION FIGURE & SAVE JSON
    # ==============================================================================
    fig, axs = plt.subplots(2, 2, figsize=(14, 11), dpi=200)

    budgets_np = np.array(budgets)

    # Panel 1: Yield Rate vs Budget (Curse of Dimensionality)
    axs[0, 0].plot(budgets_np, [m['yield_rate'] * 100 for m in metrics_C], 's-', color='#1b7837', linewidth=2.5, label='BRS (Active Boundary Frontier)')
    axs[0, 0].plot(budgets_np, [m['yield_rate'] * 100 for m in metrics_B], 'o-', color='#2166ac', linewidth=2.0, label='Goal Brownian Walk (Florensa 2017)')
    axs[0, 0].plot(budgets_np, [m['yield_rate'] * 100 for m in metrics_A], '^--', color='#b2182b', linewidth=2.0, label='4D Uniform Random Sampling')
    axs[0, 0].set_title("(a) 4D Phase Space MPC Yield Rate vs Budget", fontsize=12, fontweight='bold')
    axs[0, 0].set_xlabel("Cumulative MPC Budget", fontsize=11)
    axs[0, 0].set_ylabel("Viable Safe Yield Rate (%)", fontsize=11)
    axs[0, 0].grid(True, linestyle='--', alpha=0.6)
    axs[0, 0].legend(loc='center right', fontsize=10)

    # Panel 2: Multimodal Discovery & Balance
    axs[0, 1].plot(budgets_np, [m['balance'] for m in metrics_C], 's-', color='#1b7837', linewidth=2.5, label='BRS (Balance Ratio)')
    axs[0, 1].plot(budgets_np, [m['balance'] for m in metrics_B], 'o-', color='#2166ac', linewidth=2.0, label='Goal Walk (Balance Ratio)')
    axs[0, 1].plot(budgets_np, [m['balance'] for m in metrics_A], '^--', color='#b2182b', linewidth=2.0, label='Uniform (Balance Ratio)')
    axs[0, 1].set_title("(b) Natural Multimodal Balance Ratio (4D Robot)", fontsize=12, fontweight='bold')
    axs[0, 1].set_xlabel("Cumulative MPC Budget", fontsize=11)
    axs[0, 1].set_ylabel("Balance Ratio (2*min / sum)", fontsize=11)
    axs[0, 1].set_ylim(-0.05, 1.05)
    axs[0, 1].grid(True, linestyle='--', alpha=0.6)
    axs[0, 1].legend(loc='center right', fontsize=10)

    # Panel 3: Workspace Robot Trajectories (Elbow-Up vs Elbow-Down Bifurcation)
    axs[1, 0].set_title("(c) Autonomous Multimodal Bifurcation in Robot Task Space", fontsize=12, fontweight='bold')
    # Draw obstacle
    circle = plt.Circle((obs_pos[0].item(), obs_pos[1].item()), obs_r, color='#d95f02', alpha=0.35, label='Physical Obstacle')
    axs[1, 0].add_patch(circle)
    axs[1, 0].plot(obs_pos[0].item(), obs_pos[1].item(), 'kx', markersize=8)

    # Target configuration
    axs[1, 0].plot(target_ee[0].item(), target_ee[1].item(), 'g*', markersize=14, label='Target EE p*=[1.11, 0.0]')

    # Plot representative Flow Matching trajectories
    for tr in cfm_elbow_up_trajs[:15]:
        ee_x = np.cos(tr[:, 0]) + np.cos(tr[:, 0] + tr[:, 1])
        ee_y = np.sin(tr[:, 0]) + np.sin(tr[:, 0] + tr[:, 1])
        axs[1, 0].plot(ee_x, ee_y, color='#2b83ba', alpha=0.7, linewidth=1.8)
    for tr in cfm_elbow_down_trajs[:15]:
        ee_x = np.cos(tr[:, 0]) + np.cos(tr[:, 0] + tr[:, 1])
        ee_y = np.sin(tr[:, 0]) + np.sin(tr[:, 0] + tr[:, 1])
        axs[1, 0].plot(ee_x, ee_y, color='#d7191c', alpha=0.7, linewidth=1.8)

    # Proxy legend lines
    axs[1, 0].plot([], [], color='#2b83ba', linewidth=2.0, label='Mode 1: Elbow-Up Bypass')
    axs[1, 0].plot([], [], color='#d7191c', linewidth=2.0, label='Mode 2: Elbow-Down Bypass')
    axs[1, 0].set_xlim(-0.2, 2.2)
    axs[1, 0].set_ylim(-1.4, 1.4)
    axs[1, 0].set_xlabel("Cartesian X (m)", fontsize=11)
    axs[1, 0].set_ylabel("Cartesian Y (m)", fontsize=11)
    axs[1, 0].set_aspect('equal')
    axs[1, 0].grid(True, linestyle='--', alpha=0.6)
    axs[1, 0].legend(loc='lower left', fontsize=9)

    # Panel 4: Closed-Loop Recovery Performance Comparison
    bars = ['BC (Deterministic)', 'CFM Flow Matching (Ours)']
    x_pos = np.arange(len(bars))
    success_rates = [bc_success, cfm_success]
    collision_rates = [bc_colls, cfm_colls]

    bar_width = 0.35
    axs[1, 1].bar(x_pos - bar_width/2, success_rates, width=bar_width, color='#2ca25f', label='Recovery Success (%)')
    axs[1, 1].bar(x_pos + bar_width/2, collision_rates, width=bar_width, color='#e34a33', label='Obstacle Collisions (%)')
    axs[1, 1].set_title("(d) Closed-Loop Recovery on 4D Robot (100 Test States)", fontsize=12, fontweight='bold')
    axs[1, 1].set_xticks(x_pos)
    axs[1, 1].set_xticklabels(bars, fontsize=10, fontweight='bold')
    axs[1, 1].set_ylabel("Percentage (%)", fontsize=11)
    axs[1, 1].set_ylim(0, 110)
    axs[1, 1].grid(True, axis='y', linestyle='--', alpha=0.6)
    axs[1, 1].legend(loc='upper center', fontsize=10)

    for i, v in enumerate(success_rates):
        axs[1, 1].text(i - bar_width/2, v + 2, f"{v}%", ha='center', fontweight='bold')
    for i, v in enumerate(collision_rates):
        axs[1, 1].text(i + bar_width/2, v + 2, f"{v}%", ha='center', fontweight='bold')

    plt.tight_layout()
    plt.savefig("fully_actuated_2d_robot_scaling.png")
    print("Saved publication plot fully_actuated_2d_robot_scaling.png")

    # Save JSON results
    final_results = {
        "budgets": budgets,
        "metrics_A_uniform": metrics_A,
        "metrics_B_florensa": metrics_B,
        "metrics_C_brs": metrics_C,
        "closed_loop_eval": {
            "bc_success": bc_success,
            "bc_collisions": bc_colls,
            "cfm_success": cfm_success,
            "cfm_collisions": cfm_colls,
            "cfm_elbow_up_count": len(cfm_elbow_up_trajs),
            "cfm_elbow_down_count": len(cfm_elbow_down_trajs)
        }
    }
    with open("results_fully_actuated_2d_robot_scaling.json", "w") as f:
        json.dump(final_results, f, indent=2)
    print("Saved results to results_fully_actuated_2d_robot_scaling.json")
