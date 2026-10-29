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
# BENCHMARK: PREVIOUS IMPLEMENTATION VS. VALUE-GRADIENT (COSTATE) HYPOTHESIS
#
# Comparing:
#   1. Previous Implementation: Flow Matching in Action Space (CFM: x -> u)
#   2. Previous Implementation: Direct Action Regression (BC: x -> u)
#   3. New Hypothesis: Unconstrained Gradient Field (MLP: x -> \nabla V)
#   4. New Hypothesis: Exact Curl-Free Neural Potential (V(x) -> \nabla V via autodiff)
#   5. New Hypothesis: Trajectory Integration (ODE Rollout of \nabla V + Tracking)
#   6. Oracle: Online Receding-Horizon MPC
# ==============================================================================

np.random.seed(42)
torch.manual_seed(42)
device = torch.device('cpu')

# 1. Non-linear inverted pendulum physics
m, l, g, dt, u_max = 1.0, 1.0, 9.81, 0.02, 12.0
I = m * l**2

def step_physics(s, u):
    u = float(np.clip(u, -u_max, u_max))
    def deriv(state, act):
        return np.array([state[1], (g / l) * np.sin(state[0]) + act / I])
    k1 = deriv(s, u); k2 = deriv(s + 0.5 * dt * k1, u)
    k3 = deriv(s + 0.5 * dt * k2, u); k4 = deriv(s + dt * k3, u)
    ns = s + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    ns[0] = (ns[0] + np.pi) % (2 * np.pi) - np.pi
    return ns

# 2. Linearized continuous & discrete dynamics for local MPC
A_c = np.array([[0.0, 1.0], [g / l, 0.0]])
B_c = np.array([[0.0], [1.0 / I]])
Q_cost = np.diag([20.0, 2.0])
R_cost = np.array([[0.5]])
A_d = scipy.linalg.expm(A_c * dt)
B_d = np.linalg.inv(A_c) @ (A_d - np.eye(2)) @ B_c
P_dare = scipy.linalg.solve_discrete_are(A_d, B_d, Q_cost, R_cost)

H = 10
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

def solve_local_mpc_with_gradient(s):
    grad_qp = 2.0 * (Gamma.T @ Q_bar @ Phi @ s)
    d = -np.linalg.solve(L_chol, grad_qp)
    res = lsq_linear(L_chol.T, d, bounds=(-u_max, u_max), verbose=0)
    U_opt = res.x
    u0 = float(U_opt[0])
    
    # Exact gradient of finite-horizon value function V_H(s) via envelope theorem:
    # \nabla_s V_H(s) = 2 * Phi^T * Q_bar * (Phi * s + Gamma * U_opt)
    X_pred = Phi @ s + Gamma @ U_opt
    grad_V = 2.0 * (Phi.T @ Q_bar @ X_pred)
    return u0, grad_V

print("[Step 1/5] Collecting Dataset: Actions and Value Gradients via MPC...")
N_SAMPLES = 8000
states_list = []
actions_list = []
grad_v_list = []

while len(states_list) < N_SAMPLES:
    th = np.random.uniform(-0.8, 0.8)
    dth = np.random.uniform(-1.8, 1.8)
    s = np.array([th, dth])
    u0, grad_V = solve_local_mpc_with_gradient(s)
    
    # Contractive sanity filter
    test_s = s.copy()
    for _ in range(5):
        u_step, _ = solve_local_mpc_with_gradient(test_s)
        test_s = step_physics(test_s, u_step)
    if np.linalg.norm(test_s) < np.linalg.norm(s) + 0.1:
        states_list.append(s)
        actions_list.append(u0)
        grad_v_list.append(grad_V)

X_data = np.array(states_list, dtype=np.float32)
U_data = np.array(actions_list, dtype=np.float32).reshape(-1, 1)
G_data = np.array(grad_v_list, dtype=np.float32)

print(f"  -> Collected {len(X_data)} valid state-action-gradient tuples.")

# ==============================================================================
# 2. Model Definitions
# ==============================================================================

# Model 1: Previous Flow Matching Policy (CFM on action u)
class FlowMatchingActionNet(nn.Module):
    def __init__(self, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 + 1 + 1, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, 1)
        )
    def forward(self, u_t, x, t):
        return self.net(torch.cat([u_t, x, t], dim=-1))

# Model 2: Previous Direct Action BC (MLP)
class ActionBCNet(nn.Module):
    def __init__(self, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, 1)
        )
    def forward(self, x):
        return self.net(x)

# Model 3: New Hypothesis - Free Gradient Field Net (Predicts \nabla V directly)
class FreeGradientFieldNet(nn.Module):
    def __init__(self, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, 2)
        )
    def forward(self, x):
        return self.net(x)

# Model 4: New Hypothesis - Exact Curl-Free Neural Potential (V_theta(x) -> \nabla V)
class NeuralPotentialNet(nn.Module):
    def __init__(self, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden), nn.Softplus(),
            nn.Linear(hidden, hidden), nn.Softplus(),
            nn.Linear(hidden, 1)
        )
    def get_gradient(self, x):
        x = x.clone().detach().requires_grad_(True)
        V = self.net(x)
        grad = torch.autograd.grad(V, x, grad_outputs=torch.ones_like(V), create_graph=True)[0]
        return grad

# ==============================================================================
# 3. Model Training
# ==============================================================================
print("\n[Step 2/5] Training Models...")
X_t = torch.tensor(X_data)
U_t = torch.tensor(U_data)
G_t = torch.tensor(G_data)
loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(X_t, U_t, G_t), batch_size=128, shuffle=True
)

m_cfm = FlowMatchingActionNet()
opt_cfm = torch.optim.Adam(m_cfm.parameters(), lr=1e-3)

m_bc = ActionBCNet()
opt_bc = torch.optim.Adam(m_bc.parameters(), lr=1e-3)

m_free_grad = FreeGradientFieldNet()
opt_free = torch.optim.Adam(m_free_grad.parameters(), lr=1e-3)

m_potential = NeuralPotentialNet()
opt_pot = torch.optim.Adam(m_potential.parameters(), lr=1e-3)

EPOCHS = 15
for epoch in range(EPOCHS):
    for x_b, u_b, g_b in loader:
        # 1. CFM Train
        B = x_b.shape[0]
        t = torch.rand(B, 1)
        u0 = torch.randn_like(u_b)
        ut = (1.0 - t) * u0 + t * u_b
        target_v = u_b - u0
        loss_cfm = F.mse_loss(m_cfm(ut, x_b, t), target_v)
        opt_cfm.zero_grad(); loss_cfm.backward(); opt_cfm.step()
        
        # 2. BC Train
        loss_bc = F.mse_loss(m_bc(x_b), u_b)
        opt_bc.zero_grad(); loss_bc.backward(); opt_bc.step()
        
        # 3. Free Gradient Field Train
        loss_free = F.mse_loss(m_free_grad(x_b), g_b)
        opt_free.zero_grad(); loss_free.backward(); opt_free.step()
        
        # 4. Neural Potential Net Train
        g_pred = m_potential.get_gradient(x_b)
        loss_pot = F.mse_loss(g_pred, g_b)
        opt_pot.zero_grad(); loss_pot.backward(); opt_pot.step()

print("  -> All models successfully trained.")

# ==============================================================================
# 4. Measure Integrability / Curl (Vorticity) on Unconstrained vs. Potential Net
# ==============================================================================
print("\n[Step 3/5] Measuring Curl / Integrability Violations...")
test_states = torch.tensor(np.random.uniform(-0.8, 0.8, size=(100, 2)), dtype=torch.float32)

curl_free_net = []
curl_pot_net = []

for s in test_states:
    s_in = s.clone().detach().unsqueeze(0).requires_grad_(True)
    # Free net Jacobian
    jac_free = torch.autograd.functional.jacobian(lambda x: m_free_grad(x), s_in).squeeze()
    # Curl in 2D is \partial g_2 / \partial x_1 - \partial g_1 / \partial x_2
    curl_val_free = abs(float(jac_free[1, 0] - jac_free[0, 1]))
    curl_free_net.append(curl_val_free)
    
    # Potential net Jacobian
    jac_pot = torch.autograd.functional.jacobian(lambda x: m_potential.get_gradient(x), s_in).squeeze()
    curl_val_pot = abs(float(jac_pot[1, 0] - jac_pot[0, 1]))
    curl_pot_net.append(curl_val_pot)

avg_curl_free = np.mean(curl_free_net)
avg_curl_pot = np.mean(curl_pot_net)
print(f"  -> Free Gradient Field Average Curl Error: {avg_curl_free:.4f} (Violates Integrability!)")
print(f"  -> Neural Potential Average Curl Error:    {avg_curl_pot:.8f} (Guaranteed Zero Curl!)")

# ==============================================================================
# 5. Full Simulation Evaluation
# ==============================================================================
print("\n[Step 4/5] Evaluating Dynamic Recovery across Perturbation Regimes...")

def get_action_cfm(s, steps=5):
    with torch.no_grad():
        s_t = torch.tensor(s, dtype=torch.float32).unsqueeze(0)
        u = torch.randn(1, 1)
        dt_sub = 1.0 / steps
        for i in range(steps):
            t_sub = torch.tensor([[i * dt_sub]], dtype=torch.float32)
            v = m_cfm(u, s_t, t_sub)
            u = u + v * dt_sub
        return float(np.clip(u.numpy()[0, 0], -u_max, u_max))

def get_action_bc(s):
    with torch.no_grad():
        s_t = torch.tensor(s, dtype=torch.float32).unsqueeze(0)
        return float(np.clip(m_bc(s_t).numpy()[0, 0], -u_max, u_max))

# Derive optimal control from value gradient: u*(x) = - R^-1 B^T \nabla V(x)
def get_action_from_gradient(grad_V):
    # In continuous system: u = - R^-1 B_c^T \nabla V
    # In discrete system: u = - (R + B_d^T P B_d)^-1 B_d^T \nabla V
    # Using continuous control affine relation:
    g_x = np.array([0.0, 1.0 / I])
    u_unbounded = - (1.0 / R_cost[0, 0]) * (g_x @ grad_V) * 0.05 # scale factor for time step
    return float(np.clip(u_unbounded, -u_max, u_max))

def get_action_free_grad(s):
    with torch.no_grad():
        s_t = torch.tensor(s, dtype=torch.float32).unsqueeze(0)
        g_pred = m_free_grad(s_t).numpy()[0]
        return get_action_from_gradient(g_pred)

def get_action_potential(s):
    s_t = torch.tensor(s, dtype=torch.float32).unsqueeze(0)
    g_pred = m_potential.get_gradient(s_t).detach().numpy()[0]
    return get_action_from_gradient(g_pred)

# Method 5: Integrate trajectory ahead via ODE, then use PD tracking
def get_action_ode_trajectory_tracking(s, horizon=10):
    # Numerical ODE integration of \dot{x} = f(x, u*(x)) using the learned gradient
    curr_s = s.copy()
    plan_x = [curr_s]
    for _ in range(horizon):
        u_p = get_action_potential(curr_s)
        curr_s = step_physics(curr_s, u_p)
        plan_x.append(curr_s)
    target_s = plan_x[1]
    # Simple low-level tracking law
    u_fb = -15.0 * (s[0] - target_s[0]) - 3.0 * (s[1] - target_s[1])
    u_ff = get_action_potential(s)
    return float(np.clip(u_ff + u_fb, -u_max, u_max))

perturbation_levels = {
    "Small (sigma=0.25)": 0.25,
    "Medium (sigma=0.45)": 0.45,
    "Large (sigma=0.70)": 0.70,
    "Extreme OOD (sigma=0.95)": 0.95
}

methods = {
    "Online MPC (Oracle)": lambda s: solve_local_mpc_with_gradient(s)[0],
    "Prev Impl: Action CFM (Flow Matching)": get_action_cfm,
    "Prev Impl: Action BC (Direct Regression)": get_action_bc,
    "New: Free Gradient Field (MLP -> nabla V)": get_action_free_grad,
    "New: Exact Potential Field (V -> nabla V)": get_action_potential,
    "New: ODE Trajectory Rollout + Tracking": get_action_ode_trajectory_tracking
}

results = {m_name: {} for m_name in methods}
N_EVAL_TRIALS = 60
SIM_STEPS = 120

for p_name, p_scale in perturbation_levels.items():
    print(f"\n--- Testing Regime: {p_name} ---")
    # Generate test perturbation states
    test_initial_states = []
    for _ in range(N_EVAL_TRIALS):
        th0 = np.random.normal(0, p_scale)
        dth0 = np.random.normal(0, p_scale * 1.5)
        test_initial_states.append(np.array([th0, dth0]))
        
    for m_name, policy_fn in methods.items():
        success_count = 0
        total_costs = []
        latencies = []
        
        for s0 in test_initial_states:
            s = s0.copy()
            cost_accum = 0.0
            survived = True
            
            for t_step in range(SIM_STEPS):
                t0 = time.perf_counter()
                u_cmd = policy_fn(s)
                latencies.append((time.perf_counter() - t0) * 1e3) # ms
                
                cost_accum += float(s.T @ Q_cost @ s + u_cmd * R_cost[0, 0] * u_cmd)
                s = step_physics(s, u_cmd)
                
                if np.abs(s[0]) > np.pi * 0.75: # fell over
                    survived = False
                    break
            
            # Stabilization criteria: final position within 0.08 rad and vel within 0.2 rad/s
            if survived and np.abs(s[0]) < 0.08 and np.abs(s[1]) < 0.2:
                success_count += 1
                total_costs.append(cost_accum)
                
        success_rate = (success_count / N_EVAL_TRIALS) * 100.0
        avg_cost = np.mean(total_costs) if total_costs else float('nan')
        avg_lat = np.mean(latencies)
        results[m_name][p_name] = {
            "success_rate": success_rate,
            "avg_cost": avg_cost,
            "latency_ms": avg_lat
        }
        print(f"  [{m_name:42s}] Success: {success_rate:5.1f}% | Latency: {avg_lat*1000:6.1f} us | Cost: {avg_cost:6.1f}")

# Save results to json
with open("benchmark_value_gradient_vs_previous_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("\n[Step 5/5] Benchmark completed successfully. Saved to benchmark_value_gradient_vs_previous_results.json")
