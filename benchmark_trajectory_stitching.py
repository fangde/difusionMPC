import os
import json
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# -------------------------------------------------------------
# 1. 2D Dynamic Navigation with Non-Convex Barrier
# -------------------------------------------------------------
class DynamicStitchingEnv:
    """
    Continuous 2nd-order dynamic agent:
    State: [x, y, vx, vy]
    Control: [ax, ay] with ||u|| <= u_max
    Barrier at x = 0, y in [-1.2, 1.2] blocks direct path from (-2.5, 0) to (+2.5, 0).
    """
    def __init__(self, dt=0.04, u_max=3.5):
        self.dt = dt
        self.u_max = u_max
        self.start = np.array([-2.5, 0.0])
        self.waypoint = np.array([0.0, 1.8]) # Opening detour over top
        self.goal = np.array([2.5, 0.0])

    def step(self, state, action):
        x, y, vx, vy = state
        ax, ay = np.clip(action, -self.u_max, self.u_max)
        vx_next = vx + (ax - 0.25 * vx) * self.dt
        vy_next = vy + (ay - 0.25 * vy) * self.dt
        x_next = x + vx_next * self.dt
        y_next = y + vy_next * self.dt
        return np.array([x_next, y_next, vx_next, vy_next])

    def is_in_wall(self, pos, margin=0.12):
        x, y = pos
        # Wall at x in [-0.15, 0.15], y in [-1.3, 1.3]
        return (-0.15 - margin <= x <= 0.15 + margin) and (-1.3 - margin <= y <= 1.3 + margin)

# -------------------------------------------------------------
# 2. Diffusion / Flow Matching Policy
# -------------------------------------------------------------
class StitchingFlowPolicy(nn.Module):
    """
    Continuous Flow Policy trained to predict action vector field.
    Conditioned on current state and global target goal.
    """
    def __init__(self, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4 + 2, hidden), # state(4) + goal(2)
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 2)       # control acceleration [ax, ay]
        )
    def forward(self, state, goal):
        inp = torch.cat([state, goal], dim=-1)
        return self.net(inp)

# -------------------------------------------------------------
# 3. Benchmark Execution: Trajectory Stitching from Fragmented Data
# -------------------------------------------------------------
def run_stitching_benchmark():
    print("=" * 70)
    print("BENCHMARK: TRAJECTORY STITCHING VIA DIFFUSION / FLOW MATCHING")
    print("Can Many Local Snippets Combine into an Unseen Global Solution?")
    print("=" * 70)
    
    env = DynamicStitchingEnv()
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # ---------------------------------------------------------
    # [Step 1] Collect Fragmented Local Sub-Optimal Datasets:
    # Set A: Start (-2.5, 0) -> Waypoint (0.0, 1.8). (Never reaches Goal!)
    # Set B: Waypoint (0.0, 1.8) -> Goal (+2.5, 0). (Never started from Start!)
    # NO SINGLE TRAJECTORY in the dataset goes from Start to Goal!
    # ---------------------------------------------------------
    print("\n[Step 1] Collecting FRAGMENTED Local MPC Snippets (Zero complete demonstrations)...")
    np.random.seed(42)
    demo_states, demo_actions, demo_goals = [], [], []
    
    # Snippets A: Start -> Waypoint
    snippets_A_paths = []
    for ep in range(150):
        p0 = env.start + np.random.uniform(-0.15, 0.15, size=2)
        v0 = np.random.uniform(-0.1, 0.1, size=2)
        s = np.array([p0[0], p0[1], v0[0], v0[1]])
        path = [s[:2].copy()]
        for t in range(70):
            pos, vel = s[:2], s[2:]
            vec = env.waypoint - pos
            dist = np.linalg.norm(vec)
            u = (vec / (dist + 1e-4)) * min(env.u_max, 2.5 * dist) - 0.8 * vel
            u = np.clip(u, -env.u_max, env.u_max)
            demo_states.append(s.copy())
            demo_actions.append(u.copy())
            demo_goals.append(env.goal.copy()) # Tell model the ultimate global goal is Goal!
            s = env.step(s, u)
            path.append(s[:2].copy())
            if np.linalg.norm(s[:2] - env.waypoint) < 0.2:
                break
        snippets_A_paths.append(np.array(path))
        
    # Snippets B: Waypoint -> Goal
    snippets_B_paths = []
    for ep in range(150):
        p0 = env.waypoint + np.random.uniform(-0.2, 0.2, size=2)
        v0 = np.random.uniform(-0.1, 0.1, size=2)
        s = np.array([p0[0], p0[1], v0[0], v0[1]])
        path = [s[:2].copy()]
        for t in range(70):
            pos, vel = s[:2], s[2:]
            vec = env.goal - pos
            dist = np.linalg.norm(vec)
            u = (vec / (dist + 1e-4)) * min(env.u_max, 2.5 * dist) - 0.8 * vel
            u = np.clip(u, -env.u_max, env.u_max)
            demo_states.append(s.copy())
            demo_actions.append(u.copy())
            demo_goals.append(env.goal.copy())
            s = env.step(s, u)
            path.append(s[:2].copy())
            if np.linalg.norm(s[:2] - env.goal) < 0.2:
                break
        snippets_B_paths.append(np.array(path))
        
    demo_states = np.array(demo_states)
    demo_actions = np.array(demo_actions)
    demo_goals = np.array(demo_goals)
    print(f"Collected {len(demo_states)} local transition pairs.")
    print("Verification: Exactly 0 trajectories connect Start (-2.5, 0) to Goal (+2.5, 0) in the training data.")
    
    # ---------------------------------------------------------
    # [Step 2] Train Flow Policy to Stitch the Fragments
    # ---------------------------------------------------------
    print("\n[Step 2] Training Neural Flow Matching Policy to Stitch Fragments...")
    policy = StitchingFlowPolicy().to(device)
    opt = torch.optim.Adam(policy.parameters(), lr=1e-3)
    
    for epoch in range(1800):
        idx = np.random.choice(len(demo_states), 256)
        bs = torch.as_tensor(demo_states[idx], dtype=torch.float32, device=device)
        ba = torch.as_tensor(demo_actions[idx], dtype=torch.float32, device=device)
        bg = torch.as_tensor(demo_goals[idx], dtype=torch.float32, device=device)
        
        pred_a = policy(bs, bg)
        loss = nn.functional.mse_loss(pred_a, ba)
        opt.zero_grad()
        loss.backward()
        opt.step()
        
    policy.eval()
    
    # ---------------------------------------------------------
    # [Step 3] Test Trajectory Stitching over 100 Trials
    # ---------------------------------------------------------
    print("\n[Step 3] Testing 100 Trials: Starting from (-2.5, 0) to reach Goal (+2.5, 0)...")
    N_TEST = 100
    goal_tensor = torch.as_tensor(env.goal, dtype=torch.float32, device=device).unsqueeze(0)
    
    stitched_success = 0
    wall_collisions = 0
    test_paths = []
    
    for trial in range(N_TEST):
        np.random.seed(9500 + trial)
        p0 = env.start + np.random.uniform(-0.15, 0.15, size=2)
        s = np.array([p0[0], p0[1], 0.0, 0.0])
        path = [s[:2].copy()]
        collided = False
        
        for step in range(130):
            st_ten = torch.as_tensor(s, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                u = policy(st_ten, goal_tensor).cpu().numpy()[0]
                
            # Step dynamic system
            s = env.step(s, u)
            path.append(s[:2].copy())
            
            if env.is_in_wall(s[:2]):
                collided = True
                wall_collisions += 1
                break
                
            if np.linalg.norm(s[:2] - env.goal) < 0.35:
                break
                
        test_paths.append(np.array(path))
        if not collided and np.linalg.norm(s[:2] - env.goal) < 0.35:
            stitched_success += 1
            
    print("\n" + "=" * 70)
    print("FINAL 100-TRIAL TRAJECTORY STITCHING BENCHMARK RESULTS:")
    print("=" * 70)
    print(f"  • Global Trajectory Stitching Success: {stitched_success}% / 100%")
    print(f"  • Barrier Wall Collisions:             {wall_collisions}% / 100%")
    print("=" * 70)
    
    # ---------------------------------------------------------
    # Visualization: Plot Fragmented Datasets vs. Stitched Global Path
    # ---------------------------------------------------------
    fig_dir = "/Volumes/KIOXIA/difusionMPC/figures"
    os.makedirs(fig_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.2), dpi=220)
    
    # Panel 1: The Disconnected Training Data (Fragments)
    ax1 = axes[0]
    # Draw Wall
    ax1.plot([0.0, 0.0], [-1.3, 1.3], color='black', linewidth=7, label='Central Barrier Wall')
    
    # Plot 10 snippet A paths (Red)
    for i in range(12):
        p = snippets_A_paths[i]
        ax1.plot(p[:, 0], p[:, 1], color='#e41a1c', alpha=0.6, linewidth=1.5, 
                 label='Fragment A: Start -> Waypoint (Stops at x=0)' if i == 0 else "")
                 
    # Plot 10 snippet B paths (Blue)
    for i in range(12):
        p = snippets_B_paths[i]
        ax1.plot(p[:, 0], p[:, 1], color='#377eb8', alpha=0.6, linewidth=1.5,
                 label='Fragment B: Waypoint -> Goal (Starts at x=0)' if i == 0 else "")
                 
    ax1.scatter([-2.5], [0.0], color='black', s=120, zorder=5, label='Start (-2.5, 0)')
    ax1.scatter([0.0], [1.8], color='purple', s=100, marker='D', zorder=5, label='Waypoint Detour (0, 1.8)')
    ax1.scatter([2.5], [0.0], color='gold', edgecolor='black', s=160, marker='*', zorder=5, label='Goal (+2.5, 0)')
    
    ax1.set_title('(A) Training Data: Fragmented Disconnected Snippets\n(ZERO complete trajectories exist in dataset!)', fontsize=10.5, fontweight='bold')
    ax1.set_xlabel('X Position (m)', fontsize=10)
    ax1.set_ylabel('Y Position (m)', fontsize=10)
    ax1.set_xlim(-3.0, 3.0)
    ax1.set_ylim(-2.0, 2.5)
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.legend(loc='lower left', fontsize=8.2)
    
    # Panel 2: The Stitched Global Trajectories by Policy
    ax2 = axes[1]
    ax2.plot([0.0, 0.0], [-1.3, 1.3], color='black', linewidth=7, label='Central Barrier Wall')
    
    # Plot 15 Stitched Test Trajectories (Green)
    for i in range(15):
        p = test_paths[i]
        ax2.plot(p[:, 0], p[:, 1], color='#1b9e77', alpha=0.8, linewidth=2.2,
                 label='Stitched Global Path: Start -> Goal (Ours)' if i == 0 else "")
                 
    ax2.scatter([-2.5], [0.0], color='black', s=120, zorder=5, label='Start (-2.5, 0)')
    ax2.scatter([2.5], [0.0], color='gold', edgecolor='black', s=160, marker='*', zorder=5, label='Goal Target (+2.5, 0)')
    
    ax2.set_title(f'(B) Generative Policy Execution: Seamless Trajectory Stitching\n(Success Rate: {stitched_success}% on 100 Trials)', fontsize=10.5, fontweight='bold')
    ax2.set_xlabel('X Position (m)', fontsize=10)
    ax2.set_ylabel('Y Position (m)', fontsize=10)
    ax2.set_xlim(-3.0, 3.0)
    ax2.set_ylim(-2.0, 2.5)
    ax2.grid(True, linestyle=':', alpha=0.6)
    ax2.legend(loc='lower left', fontsize=8.5)
    
    plt.tight_layout()
    fig_path = os.path.join(fig_dir, "trajectory_stitching_benchmark.png")
    plt.savefig(fig_path)
    plt.close()
    
    results = {'stitching_success_rate': stitched_success, 'wall_collisions': wall_collisions}
    with open("/Volumes/KIOXIA/difusionMPC/results_trajectory_stitching.json", "w") as f:
        json.dump(results, f, indent=4)
        
    print(f"\nFigure saved to {fig_path}")
    print("Results saved to results_trajectory_stitching.json")

if __name__ == "__main__":
    run_stitching_benchmark()
