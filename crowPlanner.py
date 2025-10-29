import sys
import os
import numpy as np
import matplotlib.pyplot as plt

from scipy.spatial.transform import Rotation as R
from sklearn.preprocessing import normalize

import plyfile as ply
import yaml
import json

from foci.visualisation.vis_utils import ViserVis
from foci.utils.ply import extract_splat_data   
from foci.planners.planner import Planner

# ============================== ARGUMENT PARSING ==============================
# The script now expects the full path to the PLY file as the first argument
if len(sys.argv) < 2:
    print("Usage: python crowPlanner.py <path_to_splat.ply>")
    sys.exit(1)

# The path is now passed in as a command-line argument
ply_file = sys.argv[1] 

if not os.path.exists(ply_file):
    print(f"Error: PLY file not found at path: {ply_file}")
    sys.exit(1)

# Get local path (unused here, but retained)
LOCAL = os.path.dirname(os.path.abspath(__file__))

# ============================== Extract data from ply file ==============================
# read in ply file
# NOTE: extract_splat_data will now correctly open the file at the path passed from Docker
means, covs, colors, opacities = extract_splat_data(ply_file)

# The data was scaled up by 20 
means = 20 * means
covs = 20**2 * covs

radius = max(np.linalg.norm(means, axis=1)) * 1.02
robot_cov = np.eye(3) * 0.01

planner = Planner(means, covs, robot_cov, num_control_points=10, num_samples=40) 

# Load transformations
# NOTE: The path to the YAML config is still hardcoded. If this fails, it needs to be updated too.
try:
    with open("/workspace/crowPlanner.yaml", "r") as f:
        yaml_obj = yaml.safe_load(f)
except FileNotFoundError:
    print("Error: 'crowPlanner.yaml' not found in /workspace. Check your mount and file location.")
    sys.exit(1)

dp_transform = np.array(yaml_obj["dataparser_transform"])
dp_scale = yaml_obj["dataparser_scale"]
inv_dp_transform = np.linalg.inv(dp_transform) # Pre-calculate inverse

def transform_waypoint_to_relative_frame(waypoint_ned):
    """
    Transforms a waypoint from the original frame (assuming NED) to the 
    planner's relative frame (ENU, scaled). Yaw is preserved.
    """
    waypoint_enu = np.array([waypoint_ned[1], waypoint_ned[0], -waypoint_ned[2]])
    waypoint_h = np.append(waypoint_enu, 1)
    relative_point_unscaled = (dp_transform @ waypoint_h)[:3]
    relative_point_scaled = relative_point_unscaled * (20 * dp_scale)
    return relative_point_scaled.tolist()

gt_ned_waypoints = []
for i in range(5):
    try:
        # NOTE: Using the hardcoded path_1 here.
        point = yaml_obj["path_1"][f"waypoint{i}"]
        gt_ned_waypoints.append([point['x'], point['y'], point['z'], 0.0])
    except:
        # Using a more standard print for expected logic flow
        pass

if not gt_ned_waypoints:
    print("Error: No waypoints found in config file.")
    sys.exit(1)

# Transform the ground truth waypoints into the planner's relative frame
points_relative_frame = []
for wp_ned in gt_ned_waypoints:
    # Transform (x, y, z) and keep the yaw (wp_ned[3])
    rel_pos = transform_waypoint_to_relative_frame(wp_ned[:3])
    points_relative_frame.append(rel_pos + [wp_ned[3]])

print("INFO: Waypoints to plan:")
print(points_relative_frame)

# ============================== Plan in Relative Frame ==============================
solutions = []
for i in range(len(points_relative_frame) - 1):
    print(f"Planning segment {i+1}/{len(points_relative_frame) - 1}...")
    opt_curve, astar = planner.plan(points_relative_frame[i], points_relative_frame[i+1])
    solutions.append((opt_curve, astar))

# ============================== Untransform solutions into original frame ==============================
astar_original_frame = []
for sol in solutions:
    # sol[0] is the optimal curve
    for point in sol[0]:
        
        # 1. Scale Down: P_relative / (20 * dp_scale)
        scaled_point = np.array(point[:3]) / (20 * dp_scale) 
        
        # 2. Apply inv_dp_transform: P_global = inv(dp_transform) @ scaled_point
        # Convert to homogeneous and apply inv_dp_transform
        global_point_h = inv_dp_transform @ np.append(scaled_point, 1)
        
        # Extract (x, y, z) and convert to list
        astar_original_frame.append(global_point_h[:3].tolist())


astar_original_frame_ned = [(y, x, -z) for (x, y, z) in astar_original_frame]

print("\n✅ Planning Complete")

# Write to JSON
OUTPUT_FILENAME = "planned_path.json" 
output_path = os.path.join("/workspace", OUTPUT_FILENAME) 
with open(output_path, 'w') as f:
    json.dump(astar_original_frame_ned, f)
print(f"Saved planned path to {output_path}")

## Viser visualization
vis = ViserVis()
vis.add_gaussians(means, covs, color = colors,  opacity=opacities)

# Use the relative frame points for visualisation in the planner's space
vis.add_points(np.array(points_relative_frame)[:,:3], color = [1,0,0])

for i, (opt_curve, astar) in enumerate(solutions):
    # astar and opt_curve are in the relative frame
    vis.add_curve(astar[:,:3], color = [0,1,0], name = f"astar_{i}")
    vis.add_gaussian_path(opt_curve, robot_cov, planner.kinematics, color = [0,1,0], name = f"opt_curve_{i}")

vis.show()
