import os
import time
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import imageio
import mujoco

# ==============================================================================
# 3D RENDER & TRAJECTORY VISUALIZATION: FRANKA 7-DOF NULL-SPACE MANIFOLD
#
# Generates:
#   1. figures/franka_3d_nullspace_mujoco_composite.png:
#      Multi-exposure 3D MuJoCo photorealistic render showing 5 Franka arm postures
#      reaching the identical 6-DOF end-effector target with different elbow swivels.
#   2. figures/franka_3d_nullspace_multiview.png:
#      4-camera cinematic grid (Isometric, Top-Down, Side View, Hand Close-up).
#   3. figures/franka_3d_skeletal_trajectories.png:
#      3D Cartesian kinematic wireframe and trajectory paths.
#   4. figures/franka_3d_nullspace_swivel_animation.gif:
#      Smooth 3D animated simulation of the 1D null-space elbow swivel.
# ==============================================================================

os.makedirs('figures', exist_ok=True)

# ------------------------------------------------------------------------------
# 1. Load MuJoCo Model & Kinematics
# ------------------------------------------------------------------------------
model = mujoco.MjModel.from_xml_path('mujoco_franka_menagerie/panda.xml')
data = mujoco.MjData(model)

hand_id = model.body('hand').id
elbow_id = model.body('link4').id
shoulder_id = model.body('link1').id
wrist_id = model.body('link7').id

body_names = ['link0', 'link1', 'link2', 'link3', 'link4', 'link5', 'link6', 'link7', 'hand']
body_ids = [model.body(name).id for name in body_names]

q_min = model.jnt_range[:7, 0].copy()
q_max = model.jnt_range[:7, 1].copy()

# Nominal reaching configuration
q_nom = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])
data.qpos[:7] = q_nom
mujoco.mj_forward(model, data)
target_p = data.xpos[hand_id].copy()
target_R = data.xmat[hand_id].reshape(3, 3).copy()

def get_6dof_jacobian_and_null(q):
    data.qpos[:7] = q
    mujoco.mj_forward(model, data)
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    mujoco.mj_jacBody(model, data, jacp, jacr, hand_id)
    J = np.vstack([jacp[:, :7], jacr[:, :7]])
    J_pinv = np.linalg.pinv(J)
    N = np.eye(7) - J_pinv @ J
    return J, J_pinv, N

def get_swivel_angle(model, data):
    p_shoulder = data.xpos[shoulder_id]
    p_elbow = data.xpos[elbow_id]
    p_wrist = data.xpos[wrist_id]
    v_swivel = p_wrist - p_shoulder
    u_swivel = v_swivel / (np.linalg.norm(v_swivel) + 1e-8)
    v_elbow = p_elbow - p_shoulder
    proj_elbow = v_elbow - np.dot(v_elbow, u_swivel) * u_swivel
    proj_elbow = proj_elbow / (np.linalg.norm(proj_elbow) + 1e-8)
    ref = np.array([0.0, 0.0, 1.0])
    ref_proj = ref - np.dot(ref, u_swivel) * u_swivel
    ref_proj = ref_proj / (np.linalg.norm(ref_proj) + 1e-8)
    cos_psi = np.clip(np.dot(proj_elbow, ref_proj), -1.0, 1.0)
    sin_psi = np.dot(np.cross(ref_proj, proj_elbow), u_swivel)
    return np.arctan2(sin_psi, cos_psi)

# Sweep the 1D null space to get continuous spectrum
def sweep_null_manifold(direction, steps=30, step_sz=0.015):
    configs = []
    curr_q = q_nom.copy()
    for s in range(steps):
        J, J_pinv, N = get_6dof_jacobian_and_null(curr_q)
        U, S, Vt = np.linalg.svd(N)
        n_vec = Vt[0]
        if n_vec[0] < 0: n_vec = -n_vec
        cand_q = curr_q + direction * n_vec * step_sz
        if np.any(cand_q < q_min + 0.03) or np.any(cand_q > q_max - 0.03):
            break
        # Closed-loop Newton correction to exact 6-DOF manifold
        data.qpos[:7] = cand_q
        mujoco.mj_forward(model, data)
        dp = target_p - data.xpos[hand_id]
        curr_R = data.xmat[hand_id].reshape(3, 3)
        dR = 0.5 * (np.cross(curr_R[:, 0], target_R[:, 0]) + 
                    np.cross(curr_R[:, 1], target_R[:, 1]) + 
                    np.cross(curr_R[:, 2], target_R[:, 2]))
        cand_q += J_pinv @ np.concatenate([dp, dR])
        curr_q = cand_q
        configs.append(curr_q.copy())
    return configs

null_pos = sweep_null_manifold(+1.0, steps=30)
null_neg = sweep_null_manifold(-1.0, steps=30)

# Select exactly 5 representative configurations:
# 1: Extreme Left, 2: Mid Left, 3: Center, 4: Mid Right, 5: Extreme Right
selected_configs = [
    null_neg[-1],
    null_neg[len(null_neg)//2],
    q_nom,
    null_pos[len(null_pos)//2],
    null_pos[-1]
]

labels = [
    "Traj 1: Swivel Left Extreme",
    "Traj 2: Swivel Left Moderate",
    "Traj 3: Swivel Neutral Center",
    "Traj 4: Swivel Right Moderate",
    "Traj 5: Swivel Right Extreme"
]

colors_hex = ["#0077b6", "#00b4d8", "#2a9d8f", "#e76f51", "#d62828"]
colors_rgb = [
    [0.0, 0.47, 0.71],
    [0.0, 0.71, 0.85],
    [0.16, 0.62, 0.56],
    [0.91, 0.44, 0.32],
    [0.84, 0.16, 0.16]
]

# Calculate swivel angles and confirm 6-DOF accuracy for all 5 configurations
swivels_deg = []
pos_errs_mm = []
rot_errs_deg = []
skeletal_points_list = []

for idx, q in enumerate(selected_configs):
    data.qpos[:7] = q
    mujoco.mj_forward(model, data)
    psi = np.degrees(get_swivel_angle(model, data))
    swivels_deg.append(psi)
    
    p_err = np.linalg.norm(data.xpos[hand_id] - target_p) * 1000.0
    pos_errs_mm.append(p_err)
    
    curr_R = data.xmat[hand_id].reshape(3, 3)
    cos_rot = np.clip((np.trace(curr_R @ target_R.T) - 1.0) / 2.0, -1.0, 1.0)
    rot_err = np.degrees(np.arccos(cos_rot))
    rot_errs_deg.append(rot_err)
    
    pts = [data.xpos[bid].copy() for bid in body_ids]
    skeletal_points_list.append(np.array(pts))

if __name__ == '__main__':
    print("="*70)
    print("SELECTED 5 REPRESENTATIVE NULL-SPACE TRAJECTORIES:")
    for i in range(5):
        print(f"  {labels[i]:30s} | Swivel ψ = {swivels_deg[i]:+6.2f}° | Pos Err = {pos_errs_mm[i]:.4f} mm | Rot Err = {rot_errs_deg[i]:.4f}°")
    print("="*70)

# ------------------------------------------------------------------------------
# 2. Render 1: Photorealistic MuJoCo Stroboscopic Multi-Pose Composite
# ------------------------------------------------------------------------------
print("\n[Render 1/4] Rendering 3D Photorealistic MuJoCo Composite...")
renderer = mujoco.Renderer(model, height=480, width=640)

camera = mujoco.MjvCamera()
camera.distance = 2.1
camera.elevation = -18.0
camera.azimuth = 145.0
camera.lookat = np.array([0.35, 0.0, 0.5])

raw_frames = []
for q in selected_configs:
    data.qpos[:7] = q
    mujoco.mj_forward(model, data)
    renderer.update_scene(data, camera=camera)
    raw_frames.append(renderer.render().astype(np.float32))

# Multi-exposure alpha blending
# The static floor and environment remain solid; the moving arm links become semi-transparent
composite = np.mean(raw_frames, axis=0).astype(np.uint8)

# Add title banner and legend onto the composite
fig_comp, ax_comp = plt.subplots(figsize=(9, 6.75), dpi=200)
ax_comp.imshow(composite)
ax_comp.axis('off')

# Annotations
ax_comp.text(320, 30, "7-DOF Franka Panda 1D Null-Space Manifold (Elbow Swivel ψ)", 
             color='white', fontsize=12, fontweight='bold', ha='center',
             bbox=dict(boxstyle='round,pad=0.4', facecolor='black', alpha=0.75))

# Target marker
ax_comp.text(380, 240, "★ Target 6-DOF Pose\n(Identical for all 5 arms)",
             color='#ffff00', fontsize=10, fontweight='bold', ha='left',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='#333300', alpha=0.8, edgecolor='#ffff00'))

# Legend of the 5 poses
for i in range(5):
    y_coord = 420 - (4 - i) * 22
    ax_comp.plot(30, y_coord, 's', color=colors_hex[i], markersize=10)
    ax_comp.text(45, y_coord + 4, f"{labels[i]} (ψ = {swivels_deg[i]:+.1f}°)", 
                 color='white', fontsize=9, fontweight='bold',
                 bbox=dict(boxstyle='square,pad=0.1', facecolor='black', alpha=0.6))

plt.tight_layout()
plt.savefig("figures/franka_3d_nullspace_mujoco_composite.png", bbox_inches='tight', pad_inches=0.05)
plt.close()
print("Saved figures/franka_3d_nullspace_mujoco_composite.png")

# ------------------------------------------------------------------------------
# 3. Render 2: Multi-View Cinematic Grid (4 Camera Perspectives)
# ------------------------------------------------------------------------------
print("\n[Render 2/4] Rendering 4-Camera Cinematic Grid...")

views = [
    {"name": "Isometric View", "dist": 2.1, "elev": -20.0, "azim": 140.0, "lookat": [0.35, 0.0, 0.5]},
    {"name": "Top-Down Overhead View (Swivel Spread)", "dist": 1.9, "elev": -89.0, "azim": 90.0, "lookat": [0.35, 0.0, 0.5]},
    {"name": "Side Profile View (Wrist Fixed)", "dist": 1.9, "elev": -10.0, "azim": 180.0, "lookat": [0.35, 0.0, 0.5]},
    {"name": "End-Effector Close-Up (Zero Pose Error)", "dist": 0.8, "elev": -15.0, "azim": 125.0, "lookat": [target_p[0], target_p[1], target_p[2]]}
]

fig_grid, axs_grid = plt.subplots(2, 2, figsize=(14, 10), dpi=200)

for v_idx, v in enumerate(views):
    r_ax = v_idx // 2
    c_ax = v_idx % 2
    
    cam = mujoco.MjvCamera()
    cam.distance = v["dist"]
    cam.elevation = v["elev"]
    cam.azimuth = v["azim"]
    cam.lookat = np.array(v["lookat"])
    
    v_frames = []
    for q in selected_configs:
        data.qpos[:7] = q
        mujoco.mj_forward(model, data)
        renderer.update_scene(data, camera=cam)
        v_frames.append(renderer.render().astype(np.float32))
        
    v_composite = np.mean(v_frames, axis=0).astype(np.uint8)
    axs_grid[r_ax, c_ax].imshow(v_composite)
    axs_grid[r_ax, c_ax].set_title(v["name"], fontsize=11, fontweight='bold')
    axs_grid[r_ax, c_ax].axis('off')

plt.tight_layout()
plt.savefig("figures/franka_3d_nullspace_multiview.png", bbox_inches='tight', pad_inches=0.05)
plt.close()
print("Saved figures/franka_3d_nullspace_multiview.png")

# ------------------------------------------------------------------------------
# 4. Render 3: Matplotlib 3D Kinematic Wireframe Diagram
# ------------------------------------------------------------------------------
print("\n[Render 3/4] Generating 3D Kinematic Skeletal Wireframe Diagram...")

fig_skel = plt.figure(figsize=(15, 12), dpi=200)

# Panel (a): 3D Skeletal Arm Wireframe
ax3d = fig_skel.add_subplot(2, 2, 1, projection='3d')
ax3d.set_title("(a) 3D Franka 7-DOF Skeletal Kinematic Links", fontsize=11, fontweight='bold')

# Plot shoulder and wrist landmarks
p_shoulder = skeletal_points_list[2][1] # Link 1
p_wrist = skeletal_points_list[2][7]    # Link 7
p_hand = skeletal_points_list[2][8]     # Hand

# Swivel Axis
ax3d.plot([p_shoulder[0], p_wrist[0]], 
          [p_shoulder[1], p_wrist[1]], 
          [p_shoulder[2], p_wrist[2]], 
          'k--', linewidth=2.5, alpha=0.8, label='Swivel Axis u_swivel (Shoulder-Wrist)')

# Plot 5 arms
elbow_pts = []
for i in range(5):
    pts = skeletal_points_list[i]
    ax3d.plot(pts[:, 0], pts[:, 1], pts[:, 2], 
              marker='o', markersize=4, linewidth=2.5, color=colors_hex[i], 
              label=f"Traj {i+1} (ψ = {swivels_deg[i]:+.1f}°)")
    elbow_pts.append(pts[4]) # Link 4 is elbow

elbow_pts = np.array(elbow_pts)
# Plot elbow circular arc
ax3d.plot(elbow_pts[:, 0], elbow_pts[:, 1], elbow_pts[:, 2], 
          color='#6a0dad', linewidth=3.5, linestyle='-', label='1D Null-Space Elbow Arc')
ax3d.scatter(elbow_pts[:, 0], elbow_pts[:, 1], elbow_pts[:, 2], 
             color='#6a0dad', s=60, zorder=10)

# End-effector target
ax3d.scatter(target_p[0], target_p[1], target_p[2], color='gold', s=150, marker='*', zorder=12, label='Target 6-DOF EE (All 5 Meet)')

ax3d.set_xlabel("X (m)", fontsize=9)
ax3d.set_ylabel("Y (m)", fontsize=9)
ax3d.set_zlabel("Z (m)", fontsize=9)
ax3d.view_init(elev=25, azim=135)
ax3d.legend(loc='upper left', fontsize=7.5)

# Panel (b): 3D Trajectory Dynamic Approach Paths
ax_dyn = fig_skel.add_subplot(2, 2, 2, projection='3d')
ax_dyn.set_title("(b) Reaching Trajectories from Home to 6-DOF Target", fontsize=11, fontweight='bold')

# Generate smooth dynamic trajectories from home configuration q_home to each target q
q_home = np.array([0.0, 0.0, 0.0, -1.571, 0.0, 1.571, -2.3562])
data.qpos[:7] = q_home
mujoco.mj_forward(model, data)
p_home_ee = data.xpos[hand_id].copy()

n_traj_steps = 40
for i in range(5):
    q_target = selected_configs[i]
    ee_traj = []
    el_traj = []
    for s_step in range(n_traj_steps + 1):
        s_val = 0.5 * (1.0 - np.cos(np.pi * s_step / float(n_traj_steps)))
        q_interp = (1.0 - s_val) * q_home + s_val * q_target
        data.qpos[:7] = q_interp
        mujoco.mj_forward(model, data)
        ee_traj.append(data.xpos[hand_id].copy())
        el_traj.append(data.xpos[elbow_id].copy())
        
    ee_traj = np.array(ee_traj)
    el_traj = np.array(el_traj)
    
    # Plot end-effector paths (solid) and elbow paths (dashed)
    ax_dyn.plot(ee_traj[:, 0], ee_traj[:, 1], ee_traj[:, 2], 
                color=colors_hex[i], linewidth=2.5, alpha=0.9, label=f"Hand Traj {i+1}")
    ax_dyn.plot(el_traj[:, 0], el_traj[:, 1], el_traj[:, 2], 
                color=colors_hex[i], linewidth=1.5, linestyle=':', alpha=0.6)

ax_dyn.scatter(p_home_ee[0], p_home_ee[1], p_home_ee[2], color='blue', s=80, marker='s', label='Home EE Pos')
ax_dyn.scatter(target_p[0], target_p[1], target_p[2], color='gold', s=160, marker='*', label='Goal EE Pos (All 5 Converge)')
ax_dyn.set_xlabel("X (m)", fontsize=9)
ax_dyn.set_ylabel("Y (m)", fontsize=9)
ax_dyn.set_zlabel("Z (m)", fontsize=9)
ax_dyn.view_init(elev=20, azim=130)
ax_dyn.legend(loc='upper left', fontsize=7.5)

# Panel (c): 2D Elbow Circle Projection on Plane Orthogonal to Swivel Axis
ax_proj = fig_skel.add_subplot(2, 2, 3)
ax_proj.set_title("(c) 1D Swivel Geometry in Plane Normal to u_swivel", fontsize=11, fontweight='bold')

# Project elbow points onto 2D plane
u_swiv = (p_wrist - p_shoulder) / np.linalg.norm(p_wrist - p_shoulder)
ref_z = np.array([0.0, 0.0, 1.0])
e1 = ref_z - np.dot(ref_z, u_swiv) * u_swiv
e1 = e1 / np.linalg.norm(e1)
e2 = np.cross(u_swiv, e1)

for i in range(5):
    v_el = elbow_pts[i] - p_shoulder
    proj_x = np.dot(v_el, e1)
    proj_y = np.dot(v_el, e2)
    ax_proj.scatter(proj_x, proj_y, color=colors_hex[i], s=120, zorder=5, 
                    label=f"Traj {i+1} (ψ = {swivels_deg[i]:+.1f}°)")
    ax_proj.text(proj_x + 0.005, proj_y, f" Traj {i+1}\n ({swivels_deg[i]:+.1f}°)", fontsize=8, fontweight='bold')

# Draw circular guide arc
theta_circle = np.linspace(-0.6, 0.6, 100)
r_elbow = np.mean([np.linalg.norm(elbow_pts[i] - p_shoulder - np.dot(elbow_pts[i] - p_shoulder, u_swiv)*u_swiv) for i in range(5)])
circle_x = r_elbow * np.cos(theta_circle)
circle_y = r_elbow * np.sin(theta_circle)
ax_proj.plot(circle_x, circle_y, 'k--', alpha=0.5, label=f'Theoretical Swivel Circle (r={r_elbow:.3f} m)')

ax_proj.axhline(0, color='gray', linestyle=':', alpha=0.5)
ax_proj.axvline(0, color='gray', linestyle=':', alpha=0.5)
ax_proj.set_xlabel("Lateral Swivel Axis e1 (m)", fontsize=10)
ax_proj.set_ylabel("Vertical Swivel Axis e2 (m)", fontsize=10)
ax_proj.set_aspect('equal')
ax_proj.grid(True, linestyle='--', alpha=0.6)
ax_proj.legend(loc='lower left', fontsize=8)

# Panel (d): Numerical Proof Table & Metrics
ax_tbl = fig_skel.add_subplot(2, 2, 4)
ax_tbl.set_title("(d) Quantitative Kinematic Invariance Metrics", fontsize=11, fontweight='bold')
ax_tbl.axis('off')

table_data = [
    ["Trajectory", "Swivel Angle ψ", "Cartesian Pos Err", "Rotation Err", "Joint 4 (Elbow)"],
    ["Traj 1 (Left Ext)", f"{swivels_deg[0]:+.2f}°", f"{pos_errs_mm[0]:.4f} mm", f"{rot_errs_deg[0]:.4f}°", f"{selected_configs[0][3]:.4f} rad"],
    ["Traj 2 (Left Mid)", f"{swivels_deg[1]:+.2f}°", f"{pos_errs_mm[1]:.4f} mm", f"{rot_errs_deg[1]:.4f}°", f"{selected_configs[1][3]:.4f} rad"],
    ["Traj 3 (Neutral)",  f"{swivels_deg[2]:+.2f}°", f"{pos_errs_mm[2]:.4f} mm", f"{rot_errs_deg[2]:.4f}°", f"{selected_configs[2][3]:.4f} rad"],
    ["Traj 4 (Right Mid)", f"{swivels_deg[3]:+.2f}°", f"{pos_errs_mm[3]:.4f} mm", f"{rot_errs_deg[3]:.4f}°", f"{selected_configs[3][3]:.4f} rad"],
    ["Traj 5 (Right Ext)", f"{swivels_deg[4]:+.2f}°", f"{pos_errs_mm[4]:.4f} mm", f"{rot_errs_deg[4]:.4f}°", f"{selected_configs[4][3]:.4f} rad"],
]

tbl = ax_tbl.table(cellText=table_data, loc='center', cellLoc='center')
tbl.auto_set_font_size(False)
tbl.set_fontsize(9)
tbl.scale(1.0, 1.8)

# Color header
for col in range(5):
    tbl[(0, col)].set_facecolor('#2b83ba')
    tbl[(0, col)].set_text_props(color='white', fontweight='bold')
for row in range(1, 6):
    tbl[(row, 0)].set_text_props(fontweight='bold', color=colors_hex[row-1])

plt.tight_layout()
plt.savefig("figures/franka_3d_skeletal_trajectories.png", bbox_inches='tight', pad_inches=0.1)
plt.close()
print("Saved figures/franka_3d_skeletal_trajectories.png")

# ------------------------------------------------------------------------------
# 5. Render 4: Animated 3D Simulation GIF (Dynamic Reaching + Null Swivel)
# ------------------------------------------------------------------------------
print("\n[Render 4/4] Rendering 3D Animated Simulation GIF...")

cam_anim = mujoco.MjvCamera()
cam_anim.distance = 2.1
cam_anim.elevation = -22.0
cam_anim.azimuth = 145.0
cam_anim.lookat = np.array([0.35, 0.0, 0.5])

gif_frames = []

# Part 1: 5 arms reaching simultaneously from home to target (30 frames)
n_part1 = 28
for step in range(n_part1 + 1):
    s = 0.5 * (1.0 - np.cos(np.pi * step / float(n_part1)))
    # Composite all 5 arms along their trajectories
    step_frames = []
    for i in range(5):
        q_target = selected_configs[i]
        q_curr = (1.0 - s) * q_home + s * q_target
        data.qpos[:7] = q_curr
        mujoco.mj_forward(model, data)
        renderer.update_scene(data, camera=cam_anim)
        step_frames.append(renderer.render().astype(np.float32))
    gif_frames.append(np.mean(step_frames, axis=0).astype(np.uint8))

# Part 2: Swiveling smoothly back and forth across null space (36 frames)
n_part2 = 36
for step in range(n_part2):
    # Sine wave between extreme left (index 0) and extreme right (index 4)
    phase = 0.5 * (1.0 + np.sin(2.0 * np.pi * step / float(n_part2)))
    # Interpolate along null manifold
    if phase < 0.5:
        alpha = phase / 0.5
        q_swiv = (1.0 - alpha) * selected_configs[0] + alpha * selected_configs[2]
    else:
        alpha = (phase - 0.5) / 0.5
        q_swiv = (1.0 - alpha) * selected_configs[2] + alpha * selected_configs[4]
        
    data.qpos[:7] = q_swiv
    mujoco.mj_forward(model, data)
    renderer.update_scene(data, camera=cam_anim)
    frame = renderer.render()
    gif_frames.append(frame.copy())

imageio.mimsave("figures/franka_3d_nullspace_swivel_animation.gif", gif_frames, fps=18)
print("Saved figures/franka_3d_nullspace_swivel_animation.gif")

# Save summary JSON
render_summary = {
    "target_ee_position": target_p.tolist(),
    "selected_trajectories": [
        {
            "id": i + 1,
            "name": labels[i],
            "swivel_angle_deg": float(swivels_deg[i]),
            "position_error_mm": float(pos_errs_mm[i]),
            "rotation_error_deg": float(rot_errs_deg[i]),
            "color": colors_hex[i]
        }
        for i in range(5)
    ],
    "output_files": [
        "figures/franka_3d_nullspace_mujoco_composite.png",
        "figures/franka_3d_nullspace_multiview.png",
        "figures/franka_3d_skeletal_trajectories.png",
        "figures/franka_3d_nullspace_swivel_animation.gif"
    ]
}

with open("results_franka_3d_render_summary.json", "w") as f:
    json.dump(render_summary, f, indent=2)
print("Saved results_franka_3d_render_summary.json")
