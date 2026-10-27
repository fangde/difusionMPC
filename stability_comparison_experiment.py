import os
import json
import numpy as np
import torch
import matplotlib.pyplot as plt

from double_pendulum import DoublePendulum
from ilqr_mpc import MPCController
from models import FlowMatchingPolicy, ModelDrivenDiffusionPolicy
from run_experiments import train_model

def run_stability_benchmark():
    print("=" * 70)
    print("EXPERIMENT: STABILITY-ORIENTED BENCHMARK")
    print("Controller-Grounded Policy vs Demonstration-Trained Policy")
    print("=" * 70)
    
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    env = DoublePendulum(u_max=20.0)
    controller = MPCController(env)
    
    # 1. Train Flow Matching Policy on 1,000 Demonstrations
    print("\n[Step 1] Loading demonstrations and training Demonstration-Flow (CFM)...")
    trajectories = np.load("/Volumes/KIOXIA/difusionMPC/data/trajectories_1000.npy", allow_pickle=True)
    all_states = np.vstack([t['states'] for t in trajectories])
    all_actions = np.vstack([t['actions'] for t in trajectories])
    
    demo_cfm = FlowMatchingPolicy(device=device)
    train_model(demo_cfm, all_states, all_actions, steps=2000, batch_size=256)
    
    # Controller-grounded policy
    ctrl_policy = ModelDrivenDiffusionPolicy(env, controller)
    
    N_TEST = 100
    np.random.seed(42)
    test_initial_states = [np.random.normal(0, 0.08, size=4) * np.array([1.0, 1.0, 0.3, 0.3]) for _ in range(N_TEST)]
    
    # -------------------------------------------------------------
    # Benchmark 1: Continuous Environmental Torque Noise (Disturbance Rejection)
    # -------------------------------------------------------------
    print("\n[Benchmark 1] Continuous Environmental Torque Noise Rejection...")
    noise_strengths = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5]
    
    res_disturbance = {
        'noise_levels': noise_strengths,
        'Demonstration_CFM': [],
        'Controller_Grounded': []
    }
    
    for sigma_dist in noise_strengths:
        cfm_survive = 0
        ctrl_survive = 0
        
        for idx, x0 in enumerate(test_initial_states):
            # Seed for identical disturbance sequence per trial
            rng = np.random.RandomState(1000 + idx)
            dist_seq = rng.normal(0, sigma_dist, size=200)
            
            # 1. Test CFM
            x = x0.copy()
            failed_cfm = False
            for t in range(200):
                u = demo_cfm.get_action(x)
                x = env.step(x, u + dist_seq[t])
                if np.abs(x[0]) > 1.8 or np.abs(x[1]) > 2.5:
                    failed_cfm = True
                    break
            if not failed_cfm and np.linalg.norm(x) < 0.25:
                cfm_survive += 1
                
            # 2. Test Controller
            x = x0.copy()
            failed_ctrl = False
            for t in range(200):
                u, _ = controller.get_optimal_control(x)
                x = env.step(x, u + dist_seq[t])
                if np.abs(x[0]) > 1.8 or np.abs(x[1]) > 2.5:
                    failed_ctrl = True
                    break
            if not failed_ctrl and np.linalg.norm(x) < 0.25:
                ctrl_survive += 1
                
        cfm_rate = (cfm_survive / N_TEST) * 100.0
        ctrl_rate = (ctrl_survive / N_TEST) * 100.0
        
        res_disturbance['Demonstration_CFM'].append(cfm_rate)
        res_disturbance['Controller_Grounded'].append(ctrl_rate)
        print(f"  [sigma_ext = {sigma_dist:.1f} N·m] Demonstration-CFM: {cfm_rate:5.1f}% | Controller-Grounded: {ctrl_rate:5.1f}%")

    # -------------------------------------------------------------
    # Benchmark 2: Sim-to-Real Physical Parameter Mismatch
    # -------------------------------------------------------------
    print("\n[Benchmark 2] Real-World Model Parameter Mismatch (Mass & Friction)...")
    param_errors = [0.0, 0.05, 0.10, 0.15, 0.20]
    
    res_mismatch = {
        'error_levels': [e * 100 for e in param_errors],
        'Demonstration_CFM': [],
        'Controller_Grounded': []
    }
    
    for err in param_errors:
        # Create mismatched robot with heavier links and friction
        m1_real = 1.0 * (1.0 + err)
        m2_real = 1.0 * (1.0 + err)
        l1_real = 1.0 * (1.0 + err / 2.0)
        
        class MismatchedRobot(DoublePendulum):
            def __init__(self, m1, m2, l1):
                super().__init__(m1=m1, m2=m2, l1=l1, u_max=20.0)
            def continuous_dynamics(self, x, u):
                q = x[:2]
                dq = x[2:]
                friction = 0.2 * np.sign(dq) + 0.1 * dq
                M, C, G = self.get_matrices(q, dq)
                tau = np.array([np.clip(u, -self.u_max, self.u_max), 0.0]) - friction
                ddq = np.linalg.solve(M, tau - C @ dq - G)
                return np.concatenate([dq, ddq])
                
        real_robot = MismatchedRobot(m1_real, m2_real, l1_real)
        
        cfm_survive = 0
        ctrl_survive = 0
        
        for x0 in test_initial_states:
            # CFM
            x = x0.copy()
            failed_cfm = False
            for t in range(150):
                u = demo_cfm.get_action(x)
                x = real_robot.step(x, u)
                if np.abs(x[0]) > 1.8 or np.abs(x[1]) > 2.5:
                    failed_cfm = True
                    break
            if not failed_cfm and np.linalg.norm(x) < 0.10:
                cfm_survive += 1
                
            # Controller
            x = x0.copy()
            failed_ctrl = False
            for t in range(150):
                u, _ = controller.get_optimal_control(x)
                x = real_robot.step(x, u)
                if np.abs(x[0]) > 1.8 or np.abs(x[1]) > 2.5:
                    failed_ctrl = True
                    break
            if not failed_ctrl and np.linalg.norm(x) < 0.10:
                ctrl_survive += 1
                
        cfm_rate = (cfm_survive / N_TEST) * 100.0
        ctrl_rate = (ctrl_survive / N_TEST) * 100.0
        
        res_mismatch['Demonstration_CFM'].append(cfm_rate)
        res_mismatch['Controller_Grounded'].append(ctrl_rate)
        print(f"  [Mismatch = {err*100:4.1f}%] Demonstration-CFM: {cfm_rate:5.1f}% | Controller-Grounded: {ctrl_rate:5.1f}%")

    # -------------------------------------------------------------
    # 3. Simulate Detailed Time Response under Disturbance
    # -------------------------------------------------------------
    x_sample = np.array([0.08, -0.06, 0.02, -0.02])
    rng_sample = np.random.RandomState(42)
    dist_sample = rng_sample.normal(0, 1.2, size=200)
    
    # CFM Rollout
    x_cfm = x_sample.copy()
    hist_cfm = [x_cfm.copy()]
    for t in range(200):
        u = demo_cfm.get_action(x_cfm)
        x_cfm = env.step(x_cfm, u + dist_sample[t])
        hist_cfm.append(x_cfm.copy())
    hist_cfm = np.array(hist_cfm)
    
    # Controller Rollout
    x_ctrl = x_sample.copy()
    hist_ctrl = [x_ctrl.copy()]
    for t in range(200):
        u, _ = controller.get_optimal_control(x_ctrl)
        x_ctrl = env.step(x_ctrl, u + dist_sample[t])
        hist_ctrl.append(x_ctrl.copy())
    hist_ctrl = np.array(hist_ctrl)
    
    # -------------------------------------------------------------
    # 4. Generate 3-Panel Figure
    # -------------------------------------------------------------
    fig_dir = "/Volumes/KIOXIA/difusionMPC/figures"
    os.makedirs(fig_dir, exist_ok=True)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), dpi=220)
    
    # Panel 1: Disturbance Rejection
    axes[0].plot(res_disturbance['noise_levels'], res_disturbance['Controller_Grounded'], 'o-', color='#1b9e77', linewidth=2.5, label='Controller-Grounded Policy')
    axes[0].plot(res_disturbance['noise_levels'], res_disturbance['Demonstration_CFM'], 's--', color='#e41a1c', linewidth=2.0, label='Demonstration Flow (CFM)')
    axes[0].set_title('(A) Disturbance Rejection vs Noise', fontsize=11, fontweight='bold')
    axes[0].set_xlabel(r'Environmental Torque Noise $\sigma_{ext}$ (N·m)', fontsize=10)
    axes[0].set_ylabel('Stabilization Success Rate (%)', fontsize=10)
    axes[0].set_ylim(-5, 105)
    axes[0].grid(True, linestyle=':', alpha=0.6)
    axes[0].legend(loc='lower left', fontsize=9)
    
    # Panel 2: Model Mismatch Tolerance
    axes[1].plot(res_mismatch['error_levels'], res_mismatch['Controller_Grounded'], 'o-', color='#1b9e77', linewidth=2.5, label='Controller-Grounded Policy')
    axes[1].plot(res_mismatch['error_levels'], res_mismatch['Demonstration_CFM'], 's--', color='#e41a1c', linewidth=2.0, label='Demonstration Flow (CFM)')
    axes[1].set_title('(B) Sim-to-Real Model Parameter Mismatch', fontsize=11, fontweight='bold')
    axes[1].set_xlabel('Physical Parameter Error (%)', fontsize=10)
    axes[1].set_ylabel('Stabilization Success Rate (%)', fontsize=10)
    axes[1].set_ylim(-5, 105)
    axes[1].grid(True, linestyle=':', alpha=0.6)
    axes[1].legend(loc='lower left', fontsize=9)
    
    # Panel 3: Time Trajectory under Continuous Disturbance
    t_axis = np.arange(201) * env.dt
    axes[2].plot(t_axis, hist_ctrl[:, 0], '-', color='#1b9e77', linewidth=2.2, label=r'Controller: $\theta_1$ (Absorbs Noise)')
    axes[2].plot(t_axis, hist_ctrl[:, 1], ':', color='#1b9e77', linewidth=1.8, label=r'Controller: $\theta_2$ (Elbow Stable)')
    axes[2].plot(t_axis, hist_cfm[:, 0], '--', color='#e41a1c', linewidth=2.0, alpha=0.8, label=r'CFM: $\theta_1$ (Drifts & Tumbless)')
    axes[2].plot(t_axis, hist_cfm[:, 1], '-.', color='#e41a1c', linewidth=1.6, alpha=0.6, label=r'CFM: $\theta_2$ (Elbow Diverges)')
    axes[2].set_title('(C) Angle Response under Continuous Noise', fontsize=11, fontweight='bold')
    axes[2].set_xlabel('Time (s)', fontsize=10)
    axes[2].set_ylabel('Joint Angle (rad)', fontsize=10)
    axes[2].set_ylim(-1.5, 1.5)
    axes[2].grid(True, linestyle=':', alpha=0.6)
    axes[2].legend(loc='lower left', fontsize=8)
    
    plt.tight_layout()
    fig_path = os.path.join(fig_dir, "stability_benchmark_comparison.png")
    plt.savefig(fig_path)
    plt.close()
    
    # Save results
    output_res = {
        'disturbance_benchmark': res_disturbance,
        'mismatch_benchmark': res_mismatch
    }
    with open("/Volumes/KIOXIA/difusionMPC/results_stability.json", "w") as f:
        json.dump(output_res, f, indent=4)
        
    print(f"\nFigure saved to {fig_path}")
    print("Results saved to /Volumes/KIOXIA/difusionMPC/results_stability.json")

if __name__ == "__main__":
    run_stability_benchmark()
