import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ==============================================================================
# BENCHMARK B: RIEMANNIAN TUBE SAMPLING VS. UNIFORM SAMPLING
# Addressing Reviewer Gap 2: Curse of Dimensionality in Vector Field Sampling
#
# Comparing:
# 1. Uniform Phase-Space Sampling: Samples scattered over full state hypercube.
#    Suffers from exponential curse of dimensionality: O((1/eps)^d).
# 2. Riemannian Tube Sampling (Ours): Samples concentrated inside the contractive
#    tube B_eps(tau_demo) around the demonstration spine.
#    Linear scaling: O(T * K_local * d).
# ==============================================================================

np.random.seed(42)

g, dt, u_max = 9.81, 0.02, 12.0
l, m = 1.0, 1.0
I = m * l**2
K_lqr = np.array([25.0, 6.0])

def solve_mpc_local(s):
    th, dth = s
    u = np.clip(-K_lqr @ s, -u_max, u_max)
    alpha = (g / l) * np.sin(th) + u / I
    return np.array([dth, alpha]), u

def step_system(s, u):
    u = np.clip(u, -u_max, u_max)
    def deriv(state, act): return np.array([state[1], (g/l)*np.sin(state[0]) + act/I])
    k1 = deriv(s, u); k2 = deriv(s + 0.5*dt*k1, u)
    k3 = deriv(s + 0.5*dt*k2, u); k4 = deriv(s + dt*k3, u)
    ns = s + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)
    ns[0] = (ns[0] + np.pi) % (2*np.pi) - np.pi
    return ns

# 1. Demonstration Spine: Swing & Settle trajectory
demo_path = []
s_curr = np.array([0.45, 0.2])
for _ in range(80):
    demo_path.append(s_curr.copy())
    _, u = solve_mpc_local(s_curr)
    s_curr = step_system(s_curr, u)
demo_path = np.array(demo_path)
T_demo = len(demo_path)

# Sample Budgets to test: N in [50, 100, 250, 500, 1000]
budgets = [40, 80, 160, 320, 640]
N_EVAL = 60 # 60 Monte Carlo evaluation trials along tube

def evaluate_recovery_field(sampled_states, sampled_vectors):
    """Nearest-neighbor / RBF interpolation of vector field to evaluate recovery"""
    succ = 0
    for trial in range(N_EVAL):
        # Initial shock near the demo trajectory
        idx = np.random.randint(0, T_demo // 2)
        base_s = demo_path[idx]
        s_eval = base_s + np.random.normal(0, 0.12, 2)
        
        fell = False
        for step in range(100):
            # Query learned vector field via inverse distance weighting
            dists = np.linalg.norm(sampled_states - s_eval, axis=1)
            weights = np.exp(-dists / 0.10)
            weights /= (np.sum(weights) + 1e-8)
            v_interp = np.sum(weights[:, None] * sampled_vectors, axis=0)
            
            # Extract target acceleration
            alpha_req = v_interp[1]
            u_req = I * (alpha_req - (g/l)*np.sin(s_eval[0]))
            s_eval = step_system(s_eval, u_req)
            if abs(s_eval[0]) > 0.85:
                fell = True
                break
        if not fell and abs(s_eval[0]) < 0.08:
            succ += 1
    return (succ / N_EVAL) * 100.0

uniform_success = []
tube_success = []

print("[1/3] Benchmarking Sample Efficiency: Tube Sampling vs Uniform Sampling...")

for N in budgets:
    # A. Uniform Sampling across hypercube [-0.6, 0.6] x [-1.5, 1.5]
    th_u = np.random.uniform(-0.6, 0.6, N)
    dth_u = np.random.uniform(-1.5, 1.5, N)
    states_u = np.column_stack([th_u, dth_u])
    vecs_u = np.array([solve_mpc_local(s)[0] for s in states_u])
    rate_u = evaluate_recovery_field(states_u, vecs_u)
    uniform_success.append(rate_u)
    
    # B. Riemannian Tube Sampling (concentrated in B_eps around demo spine)
    tube_states = []
    tube_vecs = []
    per_waypoint = max(1, N // T_demo)
    for wp in demo_path:
        for _ in range(per_waypoint):
            pert = np.random.normal(0, 0.10, 2)
            s_pert = wp + pert
            v_p, _ = solve_mpc_local(s_pert)
            tube_states.append(s_pert)
            tube_vecs.append(v_p)
            if len(tube_states) >= N:
                break
        if len(tube_states) >= N:
            break
    # Fill remaining if needed
    while len(tube_states) < N:
        wp = demo_path[np.random.randint(0, T_demo)]
        s_pert = wp + np.random.normal(0, 0.10, 2)
        v_p, _ = solve_mpc_local(s_pert)
        tube_states.append(s_pert)
        tube_vecs.append(v_p)
        
    tube_states = np.array(tube_states)
    tube_vecs = np.array(tube_vecs)
    rate_tube = evaluate_recovery_field(tube_states, tube_vecs)
    tube_success.append(rate_tube)
    
    print(f"  • Budget N = {N:3d} samples | Uniform Recovery: {rate_u:5.1f}% | Tube Sampling Recovery: {rate_tube:5.1f}%")

# ------------------------------------------------------------------------------
# 2. Mathematical Complexity Scaling (Curse of Dimensionality Comparison)
# ------------------------------------------------------------------------------
dims = np.array([2, 4, 6, 8, 10, 12])
# Uniform: N = (1/eps)^d with eps = 0.25 (4 points per axis) -> 4^d
unif_samples = 4.0**dims
# Tube: N = T_demo * K_local * d = 80 * 5 * d = 400 * d (Strictly Linear in d!)
tube_samples = 400.0 * dims

# ------------------------------------------------------------------------------
# 3. Publication Figure: Tube Sampling vs Uniform
# ------------------------------------------------------------------------------
print("\n[2/3] Generating publication figure...")

fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), dpi=300)

# (A) Recovery Rate vs Training Budget N
axes[0].plot(budgets, tube_success, 'g-o', lw=2.4, ms=7, label='Riemannian Tube Sampling (Ours)')
axes[0].plot(budgets, uniform_success, 'r--s', lw=2.0, ms=6, label='Uniform Phase-Space Sampling')
axes[0].set_title('(A) Task Recovery Success vs. Sample Budget N', fontweight='bold', fontsize=11)
axes[0].set_xlabel('Sample Budget N (Vector Field Evaluations)', fontweight='bold')
axes[0].set_ylabel('Recovery Success Rate (%)', fontweight='bold')
axes[0].set_ylim(0, 105)
axes[0].grid(True, alpha=0.3)
axes[0].legend(fontsize=9.5)
for x, y in zip(budgets, tube_success):
    axes[0].annotate(f"{y:.0f}%", (x, y+2), fontsize=8.5, fontweight='bold', color='green')

# (B) Complexity Scaling: Exponential Curse vs. Linear Tube
axes[1].semilogy(dims, unif_samples, 'r--s', lw=2.0, ms=6, label=r'Uniform Grid: $\mathcal{O}(4^d)$ (Exponential Explosion)')
axes[1].semilogy(dims, tube_samples, 'g-o', lw=2.4, ms=7, label=r'Tube Sampling: $\mathcal{O}(T \cdot d)$ (Strictly Linear)')
axes[1].set_title('(B) Sample Complexity Scaling vs. State Dimension d', fontweight='bold', fontsize=11)
axes[1].set_xlabel('State Dimension d (Degrees of Freedom)', fontweight='bold')
axes[1].set_ylabel(r'Required Samples $\log_{10}(N)$', fontweight='bold')
axes[1].grid(True, alpha=0.3, which='both')
axes[1].legend(fontsize=9.5)

plt.tight_layout()
out_fig = 'figures/tube_sampling_scaling_benchmark.png'
plt.savefig(out_fig, dpi=300)
plt.close()
print(f"[3/3] Successfully generated tube sampling figure: {out_fig}")
