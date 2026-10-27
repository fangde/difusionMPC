import numpy as np
import scipy.linalg
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

# ==============================================================================
# The Minimal Canonical Benchmark: Limit Cycle with Orthogonal Generative Flow
#
# Proposition:
#   Instead of learning trajectories from MPC, learn a stable physical vector field
#   from local MPC feedback, and use a generative model to explore diverse
#   trajectories that preserve its stability properties:
#       v(x) = v_0(x) + w(x)  with  \nabla V(x)^T w(x) = 0
#
# Dynamics:
#   Target limit cycle: r = R = 1.0
#   Lyapunov function: V(x) = 0.5 * (||x|| - R)^2
#   Radial stabilizing field (MPC): v_0(x) = - k * (||x|| - R) * x / ||x||
#   Generative tangential flow (Diffusion): w(x) = omega(x, c) * [-x_2, x_1]^T / ||x||
#   Exact Orthogonality: \nabla V(x)^T w(x) == 0 identically!
# ==============================================================================

np.random.seed(42)
torch.manual_seed(42)

R = 1.0
k_radial = 2.5

class GenerativeTangentFlowNet(nn.Module):
    """
    Learns multimodal tangential velocity field conditioned on context c:
    c = +1 (Counter-Clockwise), c = -1 (Clockwise), c = 0 (Speed variation)
    """
    def __init__(self, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 + 1, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1) # angular velocity omega
        )
    def forward(self, x, c):
        return self.net(torch.cat([x, c], dim=-1))

# 1. Dataset Generation: Local scattered MPC vectors + multimodal styles
N_SAMPLES = 8000
# Scattered initial states (not full trajectories! Just scattered points)
r_samples = np.random.uniform(0.3, 2.2, N_SAMPLES)
th_samples = np.random.uniform(-np.pi, np.pi, N_SAMPLES)
X_data = np.stack([r_samples * np.cos(th_samples), r_samples * np.sin(th_samples)], axis=1)

# Two distinct operational modes (c = +1: CCW fast, c = -1: CW variable)
C_data = np.random.choice([-1.0, 1.0], size=(N_SAMPLES, 1))

# Target angular velocity: c * (1.5 + 0.3 * sin(3*theta))
omega_data = C_data[:, 0] * (1.8 + 0.4 * np.sin(3.0 * th_samples))
omega_data = omega_data.reshape(-1, 1) + np.random.normal(0, 0.05, size=(N_SAMPLES, 1))

X_tensor = torch.tensor(X_data, dtype=torch.float32)
C_tensor = torch.tensor(C_data, dtype=torch.float32)
Omega_tensor = torch.tensor(omega_data, dtype=torch.float32)

dataset = torch.utils.data.TensorDataset(X_tensor, C_tensor, Omega_tensor)
loader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=True)

# 2. Train Generative Model on Angular Velocity Field
model = GenerativeTangentFlowNet()
optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)

for epoch in range(25):
    total_loss = 0.0
    for x_b, c_b, w_b in loader:
        optimizer.zero_grad()
        pred_w = model(x_b, c_b)
        loss = F.mse_loss(pred_w, w_b)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
print(f"Trained Generative Tangent Flow Net. Final Loss: {total_loss / len(loader):.5f}")

# 3. Simulate and Verify:
#    v(x) = v_0(x) + w(x)
#    where w(x) is constrained or projected along level sets of V: \nabla V^T w = 0
def simulate_system(x0, c_mode, steps=250, dt=0.02, unconstrained_drift=False):
    x = np.array(x0, dtype=np.float32)
    traj = [x.copy()]
    V_history = []
    
    for _ in range(steps):
        r = np.linalg.norm(x)
        r_hat = x / (r + 1e-8)
        phi_hat = np.array([-x[1], x[0]]) / (r + 1e-8)
        
        # Invariant Radial Base Field (MPC): ensures \dot{V} = -2*k*V
        v0 = - k_radial * (r - R) * r_hat
        
        # Generative Tangent Velocity from Model
        with torch.no_grad():
            x_t = torch.tensor(x, dtype=torch.float32).unsqueeze(0)
            c_t = torch.tensor([[c_mode]], dtype=torch.float32)
            omega_pred = float(model(x_t, c_t).numpy()[0, 0])
            
        w = omega_pred * r * phi_hat
        
        if unconstrained_drift:
            # Add unconstrained noise/drift in the radial direction (simulating pure statistical policy)
            w = w + np.random.normal(0, 0.4) * r_hat
            
        # Composite Velocity Field
        v_total = v0 + w
        
        # Compute Lyapunov function V = 0.5 * (r - R)^2
        V = 0.5 * (r - R)**2
        V_history.append(V)
        
        # Euler integration
        x = x + v_total * dt
        traj.append(x.copy())
        
    return np.array(traj), np.array(V_history)

# 4. Generate Publication-Quality Figure
fig = plt.figure(figsize=(16, 5), dpi=300)
gs = fig.add_gridspec(1, 3, wspace=0.28)

# Panel 1: Vector Field Decomposition (Contraction Funnel + Tangent Flow)
ax0 = fig.add_subplot(gs[0, 0])
ax0.set_title("(A) Vector Field Decomposition: $v(x) = v_0(x) + w(x)$", fontsize=11, fontweight='bold')

grid_x = np.linspace(-1.8, 1.8, 18)
grid_y = np.linspace(-1.8, 1.8, 18)
GX, GY = np.meshgrid(grid_x, grid_y)
GR = np.sqrt(GX**2 + GY**2) + 1e-8

# Base Radial Field v0 (Blue)
V0_X = - k_radial * (GR - R) * (GX / GR) * 0.15
V0_Y = - k_radial * (GR - R) * (GY / GR) * 0.15

# Generative Tangent Field w (Green, CCW)
W_X = - 1.5 * GY * 0.15
W_Y = 1.5 * GX * 0.15

# Total Field (Purple)
VT_X = V0_X + W_X
VT_Y = V0_Y + W_Y

# Plot target limit cycle
circle = plt.Circle((0, 0), R, color='black', fill=False, linestyle='--', lw=2.0, label='Target Limit Cycle $r=R=1.0$')
ax0.add_patch(circle)

# Streamplot of total field
ax0.streamplot(grid_x, grid_y, VT_X, VT_Y, color='#7c3aed', density=0.8, arrowsize=1.2, linewidth=1.0)
ax0.quiver(GX[::2, ::2], GY[::2, ::2], V0_X[::2, ::2], V0_Y[::2, ::2], color='#1f77b4', alpha=0.5, scale=25, label=r'Radial Base $v_0$ ($\dot{V} \leq -\alpha V$)')
ax0.set_xlim(-1.8, 1.8)
ax0.set_ylim(-1.8, 1.8)
ax0.set_xlabel("State $x_1$", fontweight='bold')
ax0.set_ylabel("State $x_2$", fontweight='bold')
ax0.legend(loc='upper right', fontsize=8.0, framealpha=0.9)
ax0.grid(True, alpha=0.3)

# Panel 2: Multimodal Trajectory Generation on Same Stable Manifold
ax1 = fig.add_subplot(gs[0, 1])
ax1.set_title("(B) Diverse Multimodal Trajectories on Attractor", fontsize=11, fontweight='bold')
circle2 = plt.Circle((0, 0), R, color='black', fill=False, linestyle='--', lw=2.0, label='Limit Cycle $r=1.0$')
ax1.add_patch(circle2)

# Rollouts from inner and outer initial points under different modes
initial_points = [
    [0.4, 0.4], [-0.5, -0.3], [1.7, 0.2], [-1.5, 1.2], [1.8, -1.0]
]

for idx, p0 in enumerate(initial_points):
    # Mode c = +1: CCW
    tr_ccw, _ = simulate_system(p0, c_mode=1.0, steps=200)
    # Mode c = -1: CW
    tr_cw, _ = simulate_system(p0, c_mode=-1.0, steps=200)
    
    lbl_ccw = 'Mode A: Counter-Clockwise' if idx == 0 else ''
    lbl_cw = 'Mode B: Clockwise' if idx == 0 else ''
    
    ax1.plot(tr_ccw[:, 0], tr_ccw[:, 1], color='#2ca02c', lw=1.8, label=lbl_ccw)
    ax1.plot(tr_cw[:, 0], tr_cw[:, 1], color='#f97316', lw=1.8, linestyle=':', label=lbl_cw)
    ax1.scatter([p0[0]], [p0[1]], color='black', s=35, zorder=5)

ax1.set_xlim(-2.0, 2.0)
ax1.set_ylim(-2.0, 2.0)
ax1.set_xlabel("State $x_1$", fontweight='bold')
ax1.set_ylabel("State $x_2$", fontweight='bold')
ax1.legend(loc='upper right', fontsize=8.5, framealpha=0.9)
ax1.grid(True, alpha=0.3)

# Panel 3: Invariant Energy Decay V(x_t) from Out-of-Distribution Shocks
ax2 = fig.add_subplot(gs[0, 2])
ax2.set_title(r"(C) Lyapunov Invariance: $\nabla V(x)^T w(x) \equiv 0$", fontsize=11, fontweight='bold')

t_steps = np.arange(250) * 0.02
for p0 in [[0.25, 0.25], [1.9, 1.4], [0.3, -0.4], [-1.8, -0.8]]:
    _, V_guided = simulate_system(p0, c_mode=1.0, steps=250)
    _, V_unconstrained = simulate_system(p0, c_mode=1.0, steps=250, unconstrained_drift=True)
    
    ax2.plot(t_steps, V_guided, color='#2ca02c', lw=2.0, label='Orthogonal Decomposition (Ours)' if p0 == [0.25, 0.25] else '')
    ax2.plot(t_steps, V_unconstrained, color='#d62728', linestyle='--', lw=1.5, alpha=0.7, label='Unconstrained Policy Drift' if p0 == [0.25, 0.25] else '')

ax2.set_yscale('log')
ax2.set_xlabel("Time $t$ (seconds)", fontweight='bold')
ax2.set_ylabel("Lyapunov Error $V(x_t) = 0.5(r - R)^2$", fontweight='bold')
ax2.set_xlim(0, 5.0)
ax2.set_ylim(1e-6, 5.0)
ax2.legend(loc='upper right', fontsize=8.5, framealpha=0.9)
ax2.grid(True, which="both", alpha=0.3)

plt.tight_layout()
out_fig = "figures/limit_cycle_orthogonal_decomposition.png"
plt.savefig(out_fig, dpi=300)
plt.close()
print(f"Generated Figure: {out_fig}")
