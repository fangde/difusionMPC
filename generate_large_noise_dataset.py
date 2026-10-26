import numpy as np
import os
from double_pendulum import DoublePendulum
from ilqr_mpc import MPCController

def generate_multi_scale_dataset(total_trajectories=3000, horizon=150, seed=42):
    """
    Generates a scaled dataset of 3,000 trajectories specifically including large noise (up to sigma=0.35).
    """
    np.random.seed(seed)
    env = DoublePendulum(u_max=20.0)
    controller = MPCController(env)
    
    trajectories = []
    total_samples = 0
    
    print(f"Generating {total_trajectories} multi-scale demonstration trajectories (including large noise up to 0.35 rad)...")
    
    for i in range(total_trajectories):
        # Evenly distribute noise ranges:
        # 1/3 small noise: [0.05, 0.15]
        # 1/3 medium noise: [0.15, 0.25]
        # 1/3 large noise: [0.25, 0.35]
        if i < total_trajectories // 3:
            sigma = np.random.uniform(0.05, 0.15)
        elif i < 2 * (total_trajectories // 3):
            sigma = np.random.uniform(0.15, 0.25)
        else:
            sigma = np.random.uniform(0.25, 0.35)
            
        delta = np.random.normal(0, sigma, size=4) * np.array([1.0, 1.0, 0.5, 0.5])
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
            if t > 60 and np.linalg.norm(x) < 1e-4:
                break
                
        states = np.array(states)
        actions = np.array(actions)[:, None]
        next_states = np.array(next_states)
        
        trajectories.append({
            'states': states,
            'actions': actions,
            'next_states': next_states,
            'sigma': sigma
        })
        total_samples += len(states)
        
    print(f"Completed: {len(trajectories)} trajectories with {total_samples} total transitions.")
    return trajectories

if __name__ == "__main__":
    trajectories = generate_multi_scale_dataset(total_trajectories=3000)
    data_dir = "/Volumes/KIOXIA/difusionMPC/data"
    os.makedirs(data_dir, exist_ok=True)
    np.save(os.path.join(data_dir, "trajectories_3000_largenoise.npy"), trajectories, allow_pickle=True)
    print("Saved 3,000 multi-scale trajectories to data/trajectories_3000_largenoise.npy")
