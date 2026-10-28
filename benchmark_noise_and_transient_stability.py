import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

# ==============================================================================
# BENCHMARK A: SENSOR NOISE, TRANSIENT ESTIMATION ERROR & TORQUE CHATTER ABLATION
#
# Investigating Reviewer Gap 1 & 4:
# 1. Transient Phase (t in [0, 0.2s]): History buffer is filling up,
#    transient estimation error is non-zero (tilde_lambda = lambda - hat_lambda != 0).
# 2. Realistic Sensor Noise:
#    - Encoder noise on theta: sigma_th = 0.008 rad (~0.46 deg)
#    - Differentiated gyro noise on omega: sigma_w = 0.04 rad/s
#    - Actuator current/torque noise on u: sigma_u = 0.4 N*m
# 3. Torque Chatter Suppression:
#    - Compares Raw Neural Observer vs. EMA Filtered Robust Observer
# ==============================================================================

np.random.seed(42)
torch.manual_seed(42)

g, dt, u_max = 9.81, 0.02, 12.0
l_nom, m_nom = 1.0, 1.0
I_nom = m_nom * l_nom**2

K_lqr = np.array([25.0, 6.0])

def nominal_vector_field(s):
    th, dth = s
    u_nom = float(np.clip(-K_lqr @ s, -u_max, u_max))
    alpha = (g / l_nom) * np.sin(th) + u_nom / I_nom
    return np.array([dth, alpha]), u_nom

def step_physical_plant(s, u, m_real, b_real):
    I_real = m_real * l_nom**2
    u_eff = float(np.clip(u, -u_max, u_max))
    def deriv(state, act):
        return np.array([state[1], (g / l_nom) * np.sin(state[0]) + act / I_real - (b_real / I_real) * state[1]])
    k1 = deriv(s, u_eff)
    k2 = deriv(s + 0.5 * dt * k1, u_eff)
    k3 = deriv(s + 0.5 * dt * k2, u_eff)
    k4 = deriv(s + dt * k3, u_eff)
    ns = s + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
    ns[0] = (ns[0] + np.pi) % (2*np.pi) - np.pi
    return ns

# 1. Quick Trajectory Transformer
HIST_LEN = 10
class FastTrajectoryTransformer(nn.Module):
    def __init__(self, in_dim=3, d_model=48, nhead=4, num_layers=2):
        super().__init__()
        self.embed = nn.Linear(in_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=96, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.Linear(d_model, 24),
            nn.SiLU(),
            nn.Linear(24, 2)
        )
    def forward(self, x_seq):
        feat = self.transformer(self.embed(x_seq))
        pooled = feat.mean(dim=1)
        out = self.head(pooled)
        lambda_m = F.softplus(out[:, 0:1]) + 0.5
        b_fric = F.softplus(out[:, 1:2])
        return torch.cat([lambda_m, b_fric], dim=-1)

# Generate small dataset to initialize weights reliably
train_X, train_Y = [], []
for _ in range(1200):
    m_r = np.random.uniform(0.8, 2.5)
    b_r = np.random.uniform(0.0, 0.3)
    s_tmp = np.array([np.random.uniform(-0.4, 0.4), np.random.uniform(-0.8, 0.8)])
    seq = []
    for _ in range(HIST_LEN):
        u_tmp = float(np.random.uniform(-u_max, u_max))
        seq.append([s_tmp[0], s_tmp[1], u_tmp / u_max])
        s_tmp = step_physical_plant(s_tmp, u_tmp, m_r, b_r)
    train_X.append(seq)
    train_Y.append([m_r / m_nom, b_r])

loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(torch.tensor(train_X, dtype=torch.float32), torch.tensor(train_Y, dtype=torch.float32)),
    batch_size=64, shuffle=True
)
transformer = FastTrajectoryTransformer()
opt = torch.optim.Adam(transformer.parameters(), lr=2.5e-3)
for ep in range(12):
    for bx, by in loader:
        opt.zero_grad()
        loss = F.mse_loss(transformer(bx), by)
        loss.backward()
        opt.step()
transformer.eval()
print("[1/3] Fast Trajectory Transformer initialized.")

# ------------------------------------------------------------------------------
# 2. Run Stress Test across 3 Sensor & Filtering Regimes (50 trials each)
# ------------------------------------------------------------------------------
print("[2/3] Running 50-Trial Sensor Noise & Transient Stress Test...")

regimes = [
    ("Ideal Sensors (Zero Noise)", 0.0, 0.0, 0.0, False),
    ("Noisy Sensors (Raw Neural Obs)", 0.008, 0.04, 0.4, False),
    ("Noisy Sensors + EMA Filter (Ours)", 0.008, 0.04, 0.4, True),
]

N_TRIALS = 50
SIM_STEPS = 120

regime_results = {}

for name, sig_th, sig_w, sig_u, use_ema in regimes:
    succ_count = 0
    sat_steps_total = 0
    total_steps = 0
    chatter_metrics = [] # mean absolute diff in torque: E[|u_{t+1} - u_t|]
    peak_torques = []
    
    for trial in range(N_TRIALS):
        np.random.seed(8000 + trial)
        m_real = float(np.random.uniform(1.8, 2.5)) # +80% to +150% heavy load
        b_real = float(np.random.uniform(0.1, 0.3))
        I_real = m_real * l_nom**2
        
        s = np.array([np.random.uniform(-0.4, 0.4), np.random.uniform(-0.3, 0.3)])
        history = []
        u_prev = 0.0
        
        # EMA filter state
        hat_lambda_filt = 1.0
        hat_b_filt = 0.0
        
        trial_torques = []
        fell = False
        
        for step in range(SIM_STEPS):
            # Corrupt observations with realistic sensor noise
            th_meas = s[0] + np.random.normal(0, sig_th)
            w_meas = s[1] + np.random.normal(0, sig_w)
            u_meas = u_prev + np.random.normal(0, sig_u)
            
            history.append([th_meas, w_meas, u_meas / u_max])
            if len(history) > HIST_LEN:
                history.pop(0)
                
            # If buffer is still filling up in transient phase (t < 0.2s), pad with latest
            curr_seq = history.copy()
            while len(curr_seq) < HIST_LEN:
                curr_seq.insert(0, curr_seq[0])
                
            with torch.no_grad():
                seq_t = torch.tensor([curr_seq], dtype=torch.float32)
                pred = transformer(seq_t).numpy()[0]
                hat_lambda_raw = float(pred[0])
                hat_b_raw = float(pred[1])
                
            if use_ema:
                # Exponential Moving Average to eliminate high-frequency chatter
                alpha_ema = 0.20
                hat_lambda_filt = (1.0 - alpha_ema) * hat_lambda_filt + alpha_ema * hat_lambda_raw
                hat_b_filt = (1.0 - alpha_ema) * hat_b_filt + alpha_ema * hat_b_raw
                hat_lambda = hat_lambda_filt
                hat_b = hat_b_filt
            else:
                hat_lambda = hat_lambda_raw
                hat_b = hat_b_raw
                
            # 1. Diffeomorphic coordinate mapping: z = [theta, omega * sqrt(lambda)]
            z_nom = np.array([th_meas, w_meas * np.sqrt(hat_lambda)])
            
            # 2. Nominal flow
            v_nom, _ = nominal_vector_field(z_nom)
            alpha_nom = v_nom[1]
            
            # 3. Pullback acceleration
            alpha_pullback = alpha_nom / hat_lambda
            
            # 4. Plant tracking torque
            I_est = hat_lambda * I_nom
            u_plant = I_est * (alpha_pullback - (g / l_nom) * np.sin(th_meas)) + hat_b * w_meas
            
            total_steps += 1
            if abs(u_plant) > u_max:
                sat_steps_total += 1
            peak_torques.append(abs(u_plant))
            trial_torques.append(u_plant)
            
            u_prev = u_plant
            s = step_physical_plant(s, u_plant, m_real, b_real)
            if abs(s[0]) > 0.8:
                fell = True
                break
                
        if not fell and abs(s[0]) < 0.08:
            succ_count += 1
            
        # Compute torque chatter for this trial
        if len(trial_torques) > 1:
            diffs = np.abs(np.diff(trial_torques))
            chatter_metrics.append(np.mean(diffs))
            
    succ_rate = (succ_count / N_TRIALS) * 100.0
    feas_rate = (1.0 - (sat_steps_total / max(1, total_steps))) * 100.0
    mean_chatter = np.mean(chatter_metrics)
    mean_peak = np.mean(peak_torques)
    
    regime_results[name] = {
        "success_rate": succ_rate,
        "feasibility_rate": feas_rate,
        "chatter": mean_chatter,
        "peak_torque": mean_peak
    }

print("\n" + "=" * 85)
print("SENSOR NOISE & TRANSIENT PHASE BENCHMARK RESULTS (50 TRIALS PER REGIME):")
print("=" * 85)
print(f"{'Sensor Noise & Filter Regime':<36} | {'Success Rate':<12} | {'Feasibility':<13} | {'Torque Chatter':<14}")
print("-" * 85)
for name, data in regime_results.items():
    print(f"• {name:<34} | {data['success_rate']:5.1f}%      | {data['feasibility_rate']:5.1f}%       | {data['chatter']:5.3f} N·m/step")
print("=" * 85)

# ------------------------------------------------------------------------------
# 3. Publication Figure: Noise Sensitivity & Chatter Suppression
# ------------------------------------------------------------------------------
print("[3/3] Generating publication figure...")

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), dpi=300)

labels = ["Ideal Sensors\n(Zero Noise)", "Noisy Sensors\n(Raw Neural)", "Noisy + EMA\n(Robust Ours)"]
colors = ['#10b981', '#f59e0b', '#3b82f6']

# 1. Success Rate
s_vals = [regime_results[k]["success_rate"] for k in regime_results]
axes[0].bar(labels, s_vals, color=colors, alpha=0.85, edgecolor='black', width=0.5)
axes[0].set_title('(A) Recovery Success Rate Under Noise ↑', fontweight='bold', fontsize=11)
axes[0].set_ylabel('Success Rate (%)', fontweight='bold')
axes[0].set_ylim(0, 115)
axes[0].grid(axis='y', alpha=0.3)
for i, v in enumerate(s_vals):
    axes[0].text(i, v + 2, f"{v:.1f}%", ha='center', fontweight='bold', fontsize=9.5)

# 2. Dynamic Feasibility
f_vals = [regime_results[k]["feasibility_rate"] for k in regime_results]
axes[1].bar(labels, f_vals, color=colors, alpha=0.85, edgecolor='black', width=0.5)
axes[1].set_title('(B) Torque Feasibility (|u| ≤ u_max) ↑', fontweight='bold', fontsize=11)
axes[1].set_ylabel('Feasibility Rate (%)', fontweight='bold')
axes[1].set_ylim(0, 115)
axes[1].grid(axis='y', alpha=0.3)
for i, v in enumerate(f_vals):
    axes[1].text(i, v + 2, f"{v:.1f}%", ha='center', fontweight='bold', fontsize=9.5)

# 3. Torque Chatter E[|u_{t+1} - u_t|]
c_vals = [regime_results[k]["chatter"] for k in regime_results]
axes[2].bar(labels, c_vals, color=colors, alpha=0.85, edgecolor='black', width=0.5)
axes[2].set_title('(C) Torque Chatter E[|Δu|] (Motor Wear) ↓', fontweight='bold', fontsize=11)
axes[2].set_ylabel('Mean Torque Jitter (N·m/step)', fontweight='bold')
axes[2].grid(axis='y', alpha=0.3)
for i, v in enumerate(c_vals):
    axes[2].text(i, v + 0.04, f"{v:.3f}", ha='center', fontweight='bold', fontsize=9.5)

plt.tight_layout()
out_plot = 'figures/noise_and_transient_stability_benchmark.png'
plt.savefig(out_plot, dpi=300)
plt.close()
print(f"Successfully saved noise & transient ablation figure to {out_plot}")
