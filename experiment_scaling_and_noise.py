import os
import json
import numpy as np
import torch
import matplotlib.pyplot as plt

from double_pendulum import DoublePendulum
from ilqr_mpc import MPCController
from models import ScoreDiffusionPolicy, FlowMatchingPolicy, ModelDrivenNeuralFlowPolicy, ModelDrivenDiffusionPolicy
from run_experiments import evaluate_policy, train_model

def run_scaling_experiment():
    print("=" * 70)
    print("RUNNING BENCHMARK: HOW MODEL-DRIVEN DIFFUSION CHANGES WITH DATA & NOISE")
    print("=" * 70)
    
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Compute device: {device}")
    
    trajectories = np.load("/Volumes/KIOXIA/difusionMPC/data/trajectories_1000.npy", allow_pickle=True)
    env = DoublePendulum()
    mpc = MPCController(env)
    analytical_mpc_policy = ModelDrivenDiffusionPolicy(env, mpc)
    
    # -------------------------------------------------------------
    # Experiment 1: Scaling with Data Volume N
    # -------------------------------------------------------------
    print("\n--- Part 1: How Performance Scales with Dataset Size N (sigma = 0.15 rad) ---")
    data_sizes = [25, 50, 100, 250, 500, 1000]
    np.random.seed(123)
    N_TEST = 100
    test_perturbations_exp1 = [np.random.normal(0, 0.15, size=4) * np.array([1.0, 1.0, 0.5, 0.5]) for _ in range(N_TEST)]
    
    # Measure the analytical MPC upper bound
    oracle_rate = evaluate_policy(analytical_mpc_policy, env, test_perturbations_exp1)
    print(f"Analytical MPC Upper Bound (Physical Ceiling): {oracle_rate:.1f}%")
    
    results_scaling = {
        'data_sizes': data_sizes,
        'Model_Driven_Neural': [],
        'Flow_Matching': [],
        'Score_Diffusion': [],
        'Analytical_Ceiling': [oracle_rate] * len(data_sizes)
    }
    
    for n in data_sizes:
        print(f"\nTraining neural policies on N = {n} trajectories...")
        sub_states = np.vstack([t['states'] for t in trajectories[:n]])
        sub_actions = np.vstack([t['actions'] for t in trajectories[:n]])
        
        # 1. Model-Driven Neural Flow
        md_neural = ModelDrivenNeuralFlowPolicy(device=device)
        train_model(md_neural, sub_states, sub_actions, steps=1200, batch_size=256)
        md_rate = evaluate_policy(md_neural, env, test_perturbations_exp1)
        
        # 2. Flow Matching (CFM)
        cfm = FlowMatchingPolicy(device=device)
        train_model(cfm, sub_states, sub_actions, steps=1200, batch_size=256)
        cfm_rate = evaluate_policy(cfm, env, test_perturbations_exp1)
        
        # 3. Score Diffusion (only on subset for speed)
        if n in [50, 250, 1000]:
            score_pol = ScoreDiffusionPolicy(device=device)
            train_model(score_pol, sub_states, sub_actions, steps=1000, batch_size=256)
            score_rate = evaluate_policy(score_pol, env, test_perturbations_exp1)
        else:
            score_rate = 0.0
            
        results_scaling['Model_Driven_Neural'].append(md_rate)
        results_scaling['Flow_Matching'].append(cfm_rate)
        results_scaling['Score_Diffusion'].append(score_rate)
        
        print(f"  [N={n:4d}] Model-Driven Neural: {md_rate:.1f}% | CFM: {cfm_rate:.1f}% | Ceiling: {oracle_rate:.1f}%")
        
    # -------------------------------------------------------------
    # Experiment 2: Performance across Noise Spectrum sigma
    # -------------------------------------------------------------
    print("\n--- Part 2: How Performance Changes with Noise Severity sigma (N = 1000) ---")
    sigmas = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
    
    # Use best trained Model-Driven Neural Flow model (trained on N=1000)
    best_md_neural = md_neural
    best_cfm = cfm
    
    results_noise = {
        'sigmas': sigmas,
        'Model_Driven_Neural': [],
        'Flow_Matching': [],
        'Analytical_Ceiling': []
    }
    
    for sig in sigmas:
        np.random.seed(42 + int(sig * 100))
        test_pert = [np.random.normal(0, sig, size=4) * np.array([1.0, 1.0, 0.5, 0.5]) for _ in range(N_TEST)]
        
        rate_md = evaluate_policy(best_md_neural, env, test_pert)
        rate_cfm = evaluate_policy(best_cfm, env, test_pert)
        rate_oracle = evaluate_policy(analytical_mpc_policy, env, test_pert)
        
        results_noise['Model_Driven_Neural'].append(rate_md)
        results_noise['Flow_Matching'].append(rate_cfm)
        results_noise['Analytical_Ceiling'].append(rate_oracle)
        
        print(f"  [sigma={sig:.2f} rad] Model-Driven Neural: {rate_md:.1f}% | CFM: {rate_cfm:.1f}% | Analytical: {rate_oracle:.1f}%")

    # -------------------------------------------------------------
    # Save Results & Generate Plots
    # -------------------------------------------------------------
    fig_dir = "/Volumes/KIOXIA/difusionMPC/figures"
    os.makedirs(fig_dir, exist_ok=True)
    
    # Plot 1: Success Rate vs Dataset Size N (Active Neural Scaling!)
    plt.figure(figsize=(7.5, 4.8), dpi=200)
    plt.plot(data_sizes, results_scaling['Model_Driven_Neural'], '^-', color='#1b9e77', linewidth=2.5, label='Model-Driven Neural Flow (Trained)')
    plt.plot(data_sizes, results_scaling['Flow_Matching'], 's--', color='#7570b3', linewidth=2, label='Flow Matching (CFM)')
    plt.plot(data_sizes, results_scaling['Score_Diffusion'], 'x:', color='#d95f02', linewidth=1.5, label='Score Diffusion (DDPM)')
    plt.axhline(oracle_rate, color='#1b9e77', linestyle=':', alpha=0.7, label=f'Analytical MPC Ceiling ({oracle_rate:.0f}%)')
    
    plt.title('Sample Efficiency: How Neural Models Scale with Dataset Size N', fontsize=12, fontweight='bold')
    plt.xlabel('Number of Training Trajectories ($N$)', fontsize=11)
    plt.ylabel('Recovery Success Rate (%)', fontsize=11)
    plt.ylim(-5, 105)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(frameon=True, fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "success_rate_vs_dataset_size.png"))
    plt.close()
    
    # Plot 2: Success Rate vs Perturbation Noise sigma
    plt.figure(figsize=(7.5, 4.8), dpi=200)
    plt.plot(sigmas, results_noise['Model_Driven_Neural'], '^-', color='#1b9e77', linewidth=2.5, label='Model-Driven Neural Flow (N=1000)')
    plt.plot(sigmas, results_noise['Flow_Matching'], 's--', color='#7570b3', linewidth=2, label='Flow Matching (CFM, N=1000)')
    plt.plot(sigmas, results_noise['Analytical_Ceiling'], 'k:', linewidth=1.8, label='Analytical MPC Bound')
    
    plt.title('Robustness: Performance Under Increasing Perturbation Noise', fontsize=12, fontweight='bold')
    plt.xlabel('Perturbation Magnitude $\sigma$ (rad)', fontsize=11)
    plt.ylabel('Recovery Success Rate (%)', fontsize=11)
    plt.ylim(-5, 105)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(frameon=True, fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "success_rate_vs_perturbation.png"))
    plt.close()
    
    # Save JSON
    output_data = {
        'exp1_sample_efficiency': {
            'data_sizes': data_sizes,
            'Model_Driven_Neural': results_scaling['Model_Driven_Neural'],
            'Flow_Matching': results_scaling['Flow_Matching'],
            'Score_Diffusion': results_scaling['Score_Diffusion'],
            'Analytical_Ceiling': results_scaling['Analytical_Ceiling']
        },
        'exp2_robustness': results_noise
    }
    with open("/Volumes/KIOXIA/difusionMPC/results.json", "w") as f:
        json.dump(output_data, f, indent=4)
        
    print("\nScaling experiments finished! Updated figures and results.json saved.")

if __name__ == "__main__":
    run_scaling_experiment()
