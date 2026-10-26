import os
import json
import time
import numpy as np
import torch
import matplotlib.pyplot as plt

from double_pendulum import DoublePendulum
from ilqr_mpc import MPCController
from models import ScoreDiffusionPolicy, FlowMatchingPolicy, ModelDrivenDiffusionPolicy

def evaluate_policy(policy, env, test_perturbations, max_steps=150, eps_angle=0.08, eps_vel=0.15):
    """
    Evaluates policy over test perturbations.
    A trial is SUCCESS if the pendulum returns to upright and stays within eps thresholds.
    """
    successes = 0
    total = len(test_perturbations)
    
    for x0 in test_perturbations:
        x = x0.copy()
        failed = False
        
        for t in range(max_steps):
            u = policy.get_action(x)
            x = env.step(x, u)
            
            # Check catastrophic tumble
            if np.abs(x[0]) > 1.8 or np.abs(x[1]) > 2.5:
                failed = True
                break
                
        if not failed:
            # Check final stabilization
            if (np.abs(x[0]) < eps_angle and 
                np.abs(x[1]) < eps_angle and 
                np.abs(x[2]) < eps_vel and 
                np.abs(x[3]) < eps_vel):
                successes += 1
                
    return (successes / total) * 100.0

def train_model(policy, data_states, data_actions, steps=1500, batch_size=256):
    n_samples = len(data_states)
    for s in range(steps):
        idx = np.random.choice(n_samples, batch_size)
        loss = policy.train_step(data_states[idx], data_actions[idx])
    return policy

def run_all_experiments():
    print("=" * 70)
    print("STARTING EXPERIMENT BENCHMARK: MODEL-DRIVEN DIFFUSION VS SCORE & CFM")
    print("=" * 70)
    
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Using compute device: {device}")
    
    # 1. Load Demonstration Trajectories
    data_path = "/Volumes/KIOXIA/difusionMPC/data/trajectories_1000.npy"
    trajectories = np.load(data_path, allow_pickle=True)
    
    env = DoublePendulum()
    mpc = MPCController(env)
    model_driven_policy = ModelDrivenDiffusionPolicy(env, mpc)
    
    # Pre-generate evaluation test sets (reproducible seed)
    np.random.seed(123)
    N_TEST = 100
    
    # Experiment 1: Success Rate vs Dataset Size (at sigma = 0.15)
    print("\n--- Running Experiment 1: Success Rate vs Training Dataset Size ---")
    dataset_sizes = [50, 100, 250, 500, 1000]
    test_perturbations_exp1 = [np.random.normal(0, 0.15, size=4) * np.array([1.0, 1.0, 0.5, 0.5]) for _ in range(N_TEST)]
    
    results_exp1 = {
        'data_sizes': dataset_sizes,
        'Score_Diffusion': [],
        'Flow_Matching': [],
        'Model_Driven': []
    }
    
    # Model-Driven is evaluated zero-shot (uses physics directly)
    md_rate_exp1 = evaluate_policy(model_driven_policy, env, test_perturbations_exp1)
    print(f"Model-Driven Diffusion Baseline (Data-Free): {md_rate_exp1:.1f}%")
    
    for n_traj in dataset_sizes:
        print(f"\nTraining on {n_traj} trajectories...")
        # Pool data from first n_traj
        sub_states = np.vstack([t['states'] for t in trajectories[:n_traj]])
        sub_actions = np.vstack([t['actions'] for t in trajectories[:n_traj]])
        
        # Train Score Diffusion
        score_pol = ScoreDiffusionPolicy(device=device)
        train_model(score_pol, sub_states, sub_actions, steps=1200)
        score_rate = evaluate_policy(score_pol, env, test_perturbations_exp1)
        
        # Train Flow Matching
        cfm_pol = FlowMatchingPolicy(device=device)
        train_model(cfm_pol, sub_states, sub_actions, steps=1200)
        cfm_rate = evaluate_policy(cfm_pol, env, test_perturbations_exp1)
        
        results_exp1['Score_Diffusion'].append(score_rate)
        results_exp1['Flow_Matching'].append(cfm_rate)
        results_exp1['Model_Driven'].append(md_rate_exp1)
        
        print(f"  [N={n_traj}] Score Diffusion: {score_rate:.1f}% | Flow Matching: {cfm_rate:.1f}% | Model-Driven: {md_rate_exp1:.1f}%")

    # Experiment 2: Success Rate vs Perturbation Severity (Trained on 1000 trajectories)
    print("\n--- Running Experiment 2: Robustness vs Perturbation Severity (N=1000) ---")
    sigmas = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
    
    # Use the best models trained on 1000 trajectories
    full_states = np.vstack([t['states'] for t in trajectories])
    full_actions = np.vstack([t['actions'] for t in trajectories])
    
    best_score_pol = ScoreDiffusionPolicy(device=device)
    train_model(best_score_pol, full_states, full_actions, steps=2000)
    
    best_cfm_pol = FlowMatchingPolicy(device=device)
    train_model(best_cfm_pol, full_states, full_actions, steps=2000)
    
    results_exp2 = {
        'sigmas': sigmas,
        'Score_Diffusion': [],
        'Flow_Matching': [],
        'Model_Driven': []
    }
    
    for sig in sigmas:
        np.random.seed(42 + int(sig * 100))
        test_pert = [np.random.normal(0, sig, size=4) * np.array([1.0, 1.0, 0.5, 0.5]) for _ in range(N_TEST)]
        
        sc_rate = evaluate_policy(best_score_pol, env, test_pert)
        fm_rate = evaluate_policy(best_cfm_pol, env, test_pert)
        md_rate = evaluate_policy(model_driven_policy, env, test_pert)
        
        results_exp2['Score_Diffusion'].append(sc_rate)
        results_exp2['Flow_Matching'].append(fm_rate)
        results_exp2['Model_Driven'].append(md_rate)
        
        print(f"  [sigma={sig:.2f} rad] Score: {sc_rate:.1f}% | CFM: {fm_rate:.1f}% | Model-Driven: {md_rate:.1f}%")

    # Experiment 3: Detailed Sample Trajectory Simulation
    print("\n--- Running Experiment 3: Trajectory Comparison under Severe Perturbation ---")
    x_test_severe = np.array([0.22, -0.18, 0.1, -0.1])
    
    traj_records = {}
    for name, pol in [('Score Diffusion', best_score_pol), 
                      ('Flow Matching', best_cfm_pol), 
                      ('Model-Driven (Ours)', model_driven_policy)]:
        x = x_test_severe.copy()
        x_hist = [x.copy()]
        u_hist = []
        for _ in range(120):
            u = pol.get_action(x)
            x = env.step(x, u)
            x_hist.append(x.copy())
            u_hist.append(u)
        traj_records[name] = {
            'states': np.array(x_hist),
            'actions': np.array(u_hist)
        }

    # 4. Generate Figures
    fig_dir = "/Volumes/KIOXIA/difusionMPC/figures"
    os.makedirs(fig_dir, exist_ok=True)
    
    # Figure 1: Success Rate vs Dataset Size
    plt.figure(figsize=(7, 4.5), dpi=200)
    plt.plot(results_exp1['data_sizes'], results_exp1['Score_Diffusion'], 'o--', color='#d95f02', linewidth=2, label='Score Diffusion (DDPM)')
    plt.plot(results_exp1['data_sizes'], results_exp1['Flow_Matching'], 's--', color='#7570b3', linewidth=2, label='Flow Matching (CFM)')
    plt.plot(results_exp1['data_sizes'], results_exp1['Model_Driven'], '^-', color='#1b9e77', linewidth=2.5, label='Model-Driven Diffusion (Ours)')
    plt.title('Sample Efficiency: Success Rate vs. Number of Demonstrations', fontsize=12, fontweight='bold')
    plt.xlabel('Number of Training Trajectories ($N$)', fontsize=11)
    plt.ylabel('Recovery Success Rate (%)', fontsize=11)
    plt.ylim(-5, 105)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(frameon=True, fontsize=10)
    plt.tight_layout()
    fig1_path = os.path.join(fig_dir, "success_rate_vs_dataset_size.png")
    plt.savefig(fig1_path)
    plt.close()
    
    # Figure 2: Success Rate vs Perturbation Severity
    plt.figure(figsize=(7, 4.5), dpi=200)
    plt.plot(results_exp2['sigmas'], results_exp2['Score_Diffusion'], 'o--', color='#d95f02', linewidth=2, label='Score Diffusion (DDPM)')
    plt.plot(results_exp2['sigmas'], results_exp2['Flow_Matching'], 's--', color='#7570b3', linewidth=2, label='Flow Matching (CFM)')
    plt.plot(results_exp2['sigmas'], results_exp2['Model_Driven'], '^-', color='#1b9e77', linewidth=2.5, label='Model-Driven Diffusion (Ours)')
    plt.title('Robustness: Success Rate vs. Perturbation Severity', fontsize=12, fontweight='bold')
    plt.xlabel('Perturbation Magnitude $\sigma$ (rad)', fontsize=11)
    plt.ylabel('Recovery Success Rate (%)', fontsize=11)
    plt.ylim(-5, 105)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(frameon=True, fontsize=10)
    plt.tight_layout()
    fig2_path = os.path.join(fig_dir, "success_rate_vs_perturbation.png")
    plt.savefig(fig2_path)
    plt.close()

    # Figure 3: State Trajectories Comparison
    fig, axes = plt.subplots(3, 1, figsize=(8, 7), sharex=True, dpi=200)
    t_steps = np.arange(121) * env.dt
    
    colors = {'Score Diffusion': '#d95f02', 'Flow Matching': '#7570b3', 'Model-Driven (Ours)': '#1b9e77'}
    
    # Theta 1
    for name, data in traj_records.items():
        axes[0].plot(t_steps, data['states'][:, 0], label=name, color=colors[name], linewidth=2)
    axes[0].set_ylabel(r'$\theta_1$ (rad)', fontsize=11)
    axes[0].axhline(0, color='gray', linestyle=':')
    axes[0].grid(True, linestyle='--', alpha=0.5)
    axes[0].legend(loc='upper right', fontsize=9)
    axes[0].set_title('Closed-Loop Recovery Comparison Under Severe Perturbation', fontsize=12, fontweight='bold')

    # Theta 2
    for name, data in traj_records.items():
        axes[1].plot(t_steps, data['states'][:, 1], label=name, color=colors[name], linewidth=2)
    axes[1].set_ylabel(r'$\theta_2$ (rad)', fontsize=11)
    axes[1].axhline(0, color='gray', linestyle=':')
    axes[1].grid(True, linestyle='--', alpha=0.5)

    # Control Action u
    t_act = np.arange(120) * env.dt
    for name, data in traj_records.items():
        axes[2].plot(t_act, data['actions'], label=name, color=colors[name], linewidth=2)
    axes[2].set_ylabel('Motor Torque (N·m)', fontsize=11)
    axes[2].set_xlabel('Time (s)', fontsize=11)
    axes[2].axhline(env.u_max, color='red', linestyle='--', alpha=0.4, label='Torque Limits')
    axes[2].axhline(-env.u_max, color='red', linestyle='--', alpha=0.4)
    axes[2].grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    fig3_path = os.path.join(fig_dir, "trajectory_comparison.png")
    plt.savefig(fig3_path)
    plt.close()
    
    # Save raw results
    all_results = {
        'exp1_sample_efficiency': results_exp1,
        'exp2_robustness': results_exp2
    }
    with open("/Volumes/KIOXIA/difusionMPC/results.json", "w") as f:
        json.dump(all_results, f, indent=4)
        
    print("\nALL EXPERIMENTS COMPLETED!")
    print(f"Figures saved to {fig_dir}")
    print(f"Results saved to /Volumes/KIOXIA/difusionMPC/results.json")

if __name__ == "__main__":
    run_all_experiments()
