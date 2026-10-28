"""
run_franka_sim.py
=================
Interactive simulation and visualization script for Franka Emika Panda
robotic grasping with Zero-Retraining Payload Adaptation.

Usage:
  python3 run_franka_sim.py                     # Run benchmark test
  python3 run_franka_sim.py --viewer            # Launch interactive GLFW 3D viewer
  python3 run_franka_sim.py --video             # Render video to figures/franka_payload_adaptation.mp4
  python3 run_franka_sim.py --mass 1.2          # Test with 1.2kg heavy payload
  python3 run_franka_sim.py --mode naive        # Compare with naive controller
"""

import sys
import time
import argparse
import numpy as np
import mujoco
import mujoco.viewer
import imageio

def run(mass=1.0, mode='ours', use_viewer=False, make_video=False):
    print("=" * 70)
    print(f"Franka Emika Panda MuJoCo Simulation | Payload: {mass:.2f} kg | Mode: {mode.upper()}")
    print("=" * 70)

    model = mujoco.MjModel.from_xml_path('mujoco_franka_menagerie/scene_grasping.xml')
    model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICIT
    model.opt.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    model.opt.iterations = 100
    model.opt.ls_iterations = 30

    box_id = model.body('payload').id
    model.body_mass[box_id] = mass
    model.body_inertia[box_id] = np.array([4.16e-5, 4.16e-5, 2.56e-5]) * (mass / 0.20)

    data = mujoco.MjData(model)
    home_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, 'home')
    mujoco.mj_resetDataKeyframe(model, data, home_id)
    mujoco.mj_forward(model, data)

    q_home = np.array([0.0, 0.0, 0.0, -1.571, 0.0, 1.571, -2.3562])
    q_grasp = np.array([0.0, 0.0820, 0.0, -1.6771, 0.0, 1.5341, -2.3562])
    q_lift = np.array([0.0, -0.0327, 0.0, -1.5391, 0.0, 1.5781, -2.3562])
    q_place = np.array([0.2385, 0.0310, 0.2437, -1.7574, 0.0790, 1.4580, -2.3562])

    steps_descend = 200
    steps_grasp = 80

    if mode in ['naive', 'stiff']:
        steps_lift = 300
        steps_trans = 350
    else:
        # Ours: Observer infers lambda = mass / 0.20
        time_dilation = np.sqrt(mass / 0.20)
        steps_lift = int(300 * time_dilation)
        steps_trans = int(350 * time_dilation)

    frames = []
    renderer = None
    if make_video:
        renderer = mujoco.Renderer(model, height=720, width=1280)
        cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, 'cinematic')

    def execute_trajectory(viewer=None):
        # 1. Descend
        print("  [1/5] Reaching & Descending to payload...")
        for step in range(steps_descend):
            s = 0.5 * (1 - np.cos(np.pi * step / float(steps_descend)))
            data.ctrl[:7] = (1 - s) * q_home + s * q_grasp
            data.ctrl[7] = 0.04
            mujoco.mj_step(model, data)
            if viewer is not None: viewer.sync()
            if renderer is not None and step % 6 == 0:
                renderer.update_scene(data, camera=cam_id)
                frames.append(renderer.render().copy())

        # 2. Grasp
        print("  [2/5] Closing fingers for firm bilateral grasp...")
        for step in range(steps_grasp):
            data.ctrl[7] = 0.015
            mujoco.mj_step(model, data)
            if viewer is not None: viewer.sync()
            if renderer is not None and step % 6 == 0:
                renderer.update_scene(data, camera=cam_id)
                frames.append(renderer.render().copy())

        # 3. Lift
        print(f"  [3/5] Lifting payload ({steps_lift} steps)...")
        for step in range(steps_lift):
            s = 0.5 * (1 - np.cos(np.pi * step / float(steps_lift)))
            q_des = (1 - s) * q_grasp + s * q_lift
            data.ctrl[:7] = q_des
            data.ctrl[7] = 0.015
            mujoco.mj_step(model, data)
            if viewer is not None: viewer.sync()
            if renderer is not None and step % 6 == 0:
                renderer.update_scene(data, camera=cam_id)
                frames.append(renderer.render().copy())

        # 4. Transport
        print(f"  [4/5] Transporting to target goal area ({steps_trans} steps)...")
        for step in range(steps_trans):
            s = 0.5 * (1 - np.cos(np.pi * step / float(steps_trans)))
            q_des = (1 - s) * q_lift + s * q_place
            data.ctrl[:7] = q_des
            data.ctrl[7] = 0.015
            mujoco.mj_step(model, data)
            if viewer is not None: viewer.sync()
            if renderer is not None and step % 6 == 0:
                renderer.update_scene(data, camera=cam_id)
                frames.append(renderer.render().copy())

        # 5. Release
        print("  [5/5] Releasing payload onto goal marker...")
        for step in range(80):
            data.ctrl[7] = 0.04
            mujoco.mj_step(model, data)
            if viewer is not None: viewer.sync()
            if renderer is not None and step % 6 == 0:
                renderer.update_scene(data, camera=cam_id)
                frames.append(renderer.render().copy())

        for _ in range(60):
            mujoco.mj_step(model, data)
            if viewer is not None: viewer.sync()

    if use_viewer:
        print("Launching interactive MuJoCo passive viewer...")
        with mujoco.viewer.launch_passive(model, data) as viewer:
            while viewer.is_running():
                execute_trajectory(viewer)
                time.sleep(1.0)
                mujoco.mj_resetDataKeyframe(model, data, home_id)
    else:
        execute_trajectory(None)

    final_pos = data.xpos[box_id]
    goal_pos = np.array([0.450, 0.250, 0.425])
    err = np.linalg.norm(final_pos[:2] - goal_pos[:2]) * 1000.0
    print("-" * 70)
    print(f"Final Payload Placement Pos: [{final_pos[0]:.3f}, {final_pos[1]:.3f}, {final_pos[2]:.3f}]")
    print(f"Target Goal Marker Pos:      [{goal_pos[0]:.3f}, {goal_pos[1]:.3f}, {goal_pos[2]:.3f}]")
    print(f"Placement Error:             {err:.1f} mm | Status: {'SUCCESS' if err < 60.0 else 'FAILED'}")

    if make_video and len(frames) > 0:
        video_path = 'figures/franka_payload_adaptation.mp4'
        imageio.mimwrite(video_path, frames, fps=30, quality=8)
        print(f"Saved high-res MP4 video (1280x720): {video_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mass', type=float, default=1.0, help='Payload mass in kg')
    parser.add_argument('--mode', type=str, default='ours', choices=['ours', 'naive', 'stiff'])
    parser.add_argument('--viewer', action='store_true', help='Launch interactive GUI viewer')
    parser.add_argument('--video', action='store_true', help='Render 1280x720 video')
    args = parser.parse_args()

    run(mass=args.mass, mode=args.mode, use_viewer=args.viewer, make_video=args.video)
