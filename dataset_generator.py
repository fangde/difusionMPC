import numpy as np
import os
from double_pendulum import DoublePendulum
from ilqr_mpc import MPCController

def generate_demonstrations(num_trajectories=1000, horizon=150, max_sigma=0.20, seed=42):
    """
    Generates offline demonstration trajectories of double pendulum perturbation recovery.
    """
    np.random.seed(seed)
    env = DoublePendulum()
    controller = MPCController(env)
    
    trajectories = []
    total_samples = 0
    
    print(f"Generating {num_trajectories} demonstration trajectories...")
    for i in range(num_trajectories):
        # Sample random perturbation magnitude up to max_sigma
        sigma = np.random.uniform(0.05, max_sigma)
        
        # Perturbation on angles and angular velocities
        delta = np.random.normal(0, sigma, size=4)
        delta[2:] *= 0.5  # slightly lower initial velocity perturbations
        x = delta.copy()
        
        states = []
        actions = []
        next_states = []
        
        for t in range(horizon):
            u, _ = controller.get_optimal_control(x)
            x_next = env.step(x, u)
            
            states.append(x.copy())
            actions.append(u)
            next_states.append(x_next.copy())
            
            x = x_next.copy()
            # If successfully stabilized and stayed for long, can proceed
            if t > 50 and np.linalg.norm(x) < 1e-4:
                break
                
        states = np.array(states)
        actions = np.array(actions)[:, None]
        next_states = np.array(next_states)
        
        trajectories.append({
            'states': states,
            'actions': actions,
            'next_states': next_states
        })
        total_samples += len(states)
        
    print(f"Generated {len(trajectories)} trajectories with {total_samples} total state-action transitions.")
    return trajectories

if __name__ == "__main__":
    trajectories = generate_demonstrations(num_trajectories=1000)
    data_dir = "/Volumes/KIOXIA/difusionMPC/data"
    os.makedirs(data_dir, exist_ok=True)
    np.save(os.path.join(data_dir, "trajectories_1000.npy"), trajectories, allow_pickle=True)
    print("Dataset saved successfully.")
