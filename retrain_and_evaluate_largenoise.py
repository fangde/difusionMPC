import os
import json
import time
import numpy as np
import torch
import matplotlib.pyplot as plt

from double_pendulum import DoublePendulum
from ilqr_mpc import MPCController
from models import FlowMatchingPolicy, ScoreDiffusionPolicy, ModelDrivenDiffusionPolicy
from run_experiments import evaluate_policy, train_model

def run_large_noise_benchmark():
    print("=" * 70)
    print("BENCHMARK: IMPACT OF ADDING LARGE-NOISE DATA (N=3000, sigma up to 0.35 rad)")
    print("=" * 70)
    
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    data_path = "/Volumes/KIOXIA/difusionMPC/data/trajectories_3000_largenoise.npy"
    
    trajectories = np.load(data_path, allow_pickle=True)
    all_states = np.vstack([t['states'] for t in trajectories])
    all_actions = np.vstack([t['actions'] for t in trajectories])
    print(f"Loaded {len(trajectories)} trajectories with {len(all_states)} total transitions.")
    
    env = DoublePendulum(u_max=20.0)
    mpc = MPCController(env)
    model_driven_policy = ModelDrivenDiffusionPolicy(env, mpc)
    
    # Train Flow Matching on scaled dataset
    print("\nTraining Flow Matching on 3,000 large-noise trajectories...")
    cfm_scaled = FlowMatchingPolicy(device=device)
    train_model(cfm_scaled, all_states, all_actions, steps=2500, batch_size=256)
    
    # Evaluate across noise spectrum
    sigmas = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
    N_TEST = 100
    
    # Load previous 1000-demo results for direct comparison
    with open("/Volumes/KIOXIA/difusionMPC/results.json", "r") as f:
        old_results = json.load(f)
    old_cfm = old_results['exp2_robustness']['Flow_Matching']
    
    results = {
        'sigmas': sigmas,
        'Flow_Matching_old_1000': old_cfm,
        'Flow_Matching_added_3000': [],
        'Model_Driven': []
    }
    
    print("\nEvaluating robustness across noise levels...")
    for sig in sigmas:
        np.random.seed(42 + int(sig * 100))
        test_pert = [np.random.normal(0, sig, size=4) * np.array([1.0, 1.0, 0.5, 0.5]) for _ in range(N_TEST)]
        
        cfm_rate = evaluate_policy(cfm_scaled, env, test_pert)
        md_rate = evaluate_policy(model_driven_policy, env, test_pert)
        
        results['Flow_Matching_added_3000'].append(cfm_rate)
        results['Model_Driven'].append(md_rate)
        
        print(f"  [sigma={sig:.2f} rad] CFM (Old 1000): {old_cfm[sigmas.index(sig)]:.1f}% -> CFM (Added Data 3000): {cfm_rate:.1f}% | Model-Driven: {md_rate:.1f}%")
        
    # Save results
    with open("/Volumes/KIOXIA/difusionMPC/results_largenoise.json", "w") as f:
        json.dump(results, f, indent=4)
        
    # Plot comparison
    fig_dir = "/Volumes/KIOXIA/difusionMPC/figures"
    plt.figure(figsize=(7.5, 4.8), dpi=200)
    plt.plot(sigmas, results['Flow_Matching_old_1000'], 's--', color='#984ea3', linewidth=2, label='CFM (N=1000, Low Noise Data)')
    plt.plot(sigmas, results['Flow_Matching_added_3000'], 'o-', color='#377eb8', linewidth=2.5, label='CFM (N=3000, Added Large Noise Data)')
    plt.plot(sigmas, results['Model_Driven'], '^-', color='#4daf4a', linewidth=2.5, label='Model-Driven Diffusion (Ours, Data-Free)')
    
    plt.title('Robustness Under Large Noise: Effect of Adding Large-Perturbation Data', fontsize=12, fontweight='bold')
    plt.xlabel('Perturbation Magnitude $\sigma$ (rad)', fontsize=11)
    plt.ylabel('Recovery Success Rate (%)', fontsize=11)
    plt.ylim(-5, 105)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(frameon=True, fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "robustness_comparison_added_data.png"))
    plt.close()
    
    print("\nBenchmark finished and plot saved to figures/robustness_comparison_added_data.png")

if __name__ == "__main__":
    run_large_noise_benchmark()
