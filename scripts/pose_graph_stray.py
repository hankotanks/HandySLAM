import os
import sys
import numpy as np
import open3d as o3d
import scipy
import cv2
import tqdm
import pickle
# import networkx as nx
# import matplotlib.pyplot as plt

CONFIDENCE_THRESHOLD = 2

if(len(sys.argv) != 2):
    print(f"Usage: {sys.argv[0]} <scene_path>")
    exit(1)

path_scene = os.path.abspath(sys.argv[1])

path_odom = os.path.join(path_scene, "odometry.csv")
if(not os.path.exists(path_odom)): raise IOError(f"Failed to find {path_odom}")

path_traj = os.path.join(path_scene, "trajectory.txt")
if(not os.path.exists(path_traj)): raise IOError(f"Failed to find {path_traj}")

path_camera_matrix = os.path.join(path_scene, "camera_matrix.csv")
if(not os.path.exists(path_camera_matrix)): raise IOError(f"Failed to find {path_camera_matrix}")

intrinsics = np.loadtxt(path_camera_matrix, delimiter = ',')
if intrinsics.shape != (3, 3): raise IOERROR(f"{path_camera_matrix} was malformed")

FX, FY = intrinsics[0, 0], intrinsics[1, 1]
CX, CY = intrinsics[0, 2], intrinsics[1, 2]

path_color = os.path.join(path_scene, "rgb.mp4")
if(not os.path.exists(path_traj)): raise IOError(f"Failed to find {path_color}")

path_depth = os.path.join(path_scene, "depth")
if(not os.path.exists(path_depth)): raise IOError(f"Failed to find {path_depth}")

STAMPS = np.loadtxt(path_odom, delimiter = ",", skiprows = 1, usecols = [0, 1])
STAMPS = STAMPS[np.argsort(STAMPS[:, 0])]

for i in range(0, STAMPS.shape[1]):
    if(not os.path.exists(os.path.join(path_depth, f"{STAMPS[i, 1].astype(int):06d}.png"))):
        raise IOError(f"Failed to find {path_depth}")

path_confidence = os.path.join(path_scene, "confidence")
if(not os.path.exists(path_confidence)): raise IOError(f"Failed to find {path_confidence}")

for i in range(0, STAMPS.shape[1]):
    if(not os.path.exists(os.path.join(path_confidence, f"{STAMPS[i, 1].astype(int):06d}.png"))):
        raise IOError(f"Failed to find {path_confidence}")

CAP = cv2.VideoCapture(path_color)
if not CAP.isOpened(): raise IOError(f"Could not open video {path_color}")

CAP_W = int(CAP.get(cv2.CAP_PROP_FRAME_WIDTH))
CAP_H = int(CAP.get(cv2.CAP_PROP_FRAME_HEIGHT))

class Frame:
    def __init__(self, t, x, y, z, qx, qy, qz, qw):
        self.t = float(t)
        # parse pose
        q = np.array([float(qx), float(qy), float(qz), float(qw)])
        rot = scipy.spatial.transform.Rotation.from_quat(q)
        self.pose = np.eye(4)
        self.pose[0:3, 3] = np.array([float(x), float(y), float(z)])
        self.pose[0:3, 0:3] = rot.as_matrix()
        # find corresponding frame of imagery
        row = np.argmin(np.abs(STAMPS[:, 0] - self.t))
        self.frame_idx = STAMPS[row, 1].astype(int)
        self.frame_offset = np.abs(STAMPS[row, 0] - self.t)
        if self.frame_offset > 1e-5: 
            raise Exception(f"Temporal offset of frame was too large ({self.frame_offset})")
        # load rgb frame
        CAP.set(cv2.CAP_PROP_POS_FRAMES, self.frame_idx)
        ret, frame_color = CAP.read()
        if not ret: raise IOError(f"Could not read color frame {self.frame_idx} of {path_color}")
        frame_color = frame_color.astype(np.float32)
        # load depthmap
        path_depth_frame = os.path.join(path_depth, f"{self.frame_idx:06d}.png")
        frame_depth = cv2.imread(path_depth_frame, cv2.IMREAD_UNCHANGED)
        if frame_depth is None: raise IOError(f"Could not read depthmap {self.frame_idx} of {path_depth_frame}")
        frame_depth = frame_depth.astype(np.float32) / 1000.0;
        # load confidence map
        path_confidence_frame = os.path.join(path_confidence, f"{self.frame_idx:06d}.png")
        frame_confidence = cv2.imread(path_confidence_frame, cv2.IMREAD_UNCHANGED)
        if frame_confidence is None: raise IOError(f"Could not read depthmap {self.frame_idx} of {path_confidence_frame}")
        frame_confidence = frame_confidence.astype(np.float32)
        # project depthmap to point cloud
        frame_depth_rows, frame_depth_cols = frame_depth.shape
        frame_depth_fx = FX * frame_depth_cols / CAP_W
        frame_depth_fy = FY * frame_depth_rows / CAP_H
        frame_depth_cx = CX * frame_depth_cols / CAP_W
        frame_depth_cy = CY * frame_depth_rows / CAP_H
        u, v = np.meshgrid(np.arange(frame_depth_cols), np.arange(frame_depth_rows))
        u = u.flatten()
        v = v.flatten()
        z = frame_depth.flatten()
        z_filter = (z > 0) & (frame_confidence.flatten() >= CONFIDENCE_THRESHOLD)
        u = u[z_filter]
        v = v[z_filter]
        z = z[z_filter]
        X = (u - frame_depth_cx) * z / frame_depth_fx
        Y = (v - frame_depth_cy) * z / frame_depth_fy
        Z = z
        self.points = np.transpose(np.vstack((X, Y, Z)))

path_blob = os.path.join(path_scene, "frames.pkl")

FRAMES = None
if(os.path.exists(path_blob)):
    with open(path_blob, "rb") as f:
        FRAMES = pickle.load(f)
else:
    FRAMES = {}
    with open(path_traj, "r") as f:
        lines = f.readlines()
        for line in tqdm.tqdm(lines, "projecting"):
            parts = line.strip().split()
            if len(parts) != 8: continue
            frame = None
            try:
                frame = Frame(*parts)
            except Exception as e: 
                print(f"Failed to process frame: {e}"); 
            if frame is not None:
                FRAMES[frame.frame_idx] = frame

    with open(path_blob, "wb") as f:
        pickle.dump(FRAMES, f)

if FRAMES is None: 
    raise Exception("Failed to process scene")

path_edges = os.path.join(path_scene, "edges.txt")
if not os.path.exists(path_edges): raise IOError(f"Failed to find {path_edges}")

edges = np.loadtxt(path_edges, usecols = [0, 1])
if edges.shape[1] != 2: raise IOError(f"{path_edges} was malformed")

edges_closures = np.argmin(np.abs(edges[:, :, None] - STAMPS[:, 0][None, None, :]), axis = 2)

edges_adjacent = np.array([[i, i+1] for i in range(edges_closures.min(), max(FRAMES.keys()))])
edges_closures = edges_closures[np.abs(edges_closures[:, 0] - edges_closures[:, 1]) != 1]

""" # visualize edges
G = nx.Graph()
G.add_edges_from([tuple(row) for row in edges_adjacent])
G.add_edges_from([tuple(row) for row in edges_closures])
pos = nx.spring_layout(G, seed=42)
nx.draw(G, pos, with_labels=True, node_size=300, node_color='skyblue', edge_color='gray')
plt.show()
exit(0)
"""

print(max(list(FRAMES.keys())))

VOXEL_SIZE = 0.02 # 2cm
MAX_CORRESPONDENCE_DISTANCE = VOXEL_SIZE * 1.5

POSE_GRAPH = o3d.pipelines.registration.PoseGraph()

frame_map = {}
frame_count = 0
for frame_idx, frame in FRAMES.items():
    POSE_GRAPH.nodes.append(o3d.pipelines.registration.PoseGraphNode(np.linalg.inv(frame.pose)))
    frame_map[frame_idx] = frame_count
    frame_count += 1

def append_edges(edges, uncertain):
    for edge in edges:
        frame_fst = FRAMES[edge[0]]
        frame_snd = FRAMES[edge[1]]
        frame_fst_idx = frame_map[frame_fst.frame_idx]
        frame_snd_idx = frame_map[frame_snd.frame_idx]
        POSE_GRAPH.edges.append(o3d.pipelines.registration.PoseGraphEdge(
            frame_fst_idx, frame_snd_idx, 
            np.linalg.inv(frame_snd.pose) @ frame_fst.pose, 
            np.identity(6), uncertain = uncertain))

append_edges(edges_adjacent, False)
append_edges(edges_closures, True)

options = o3d.pipelines.registration.GlobalOptimizationOption(
    max_correspondence_distance = MAX_CORRESPONDENCE_DISTANCE,
    edge_prune_threshold = 0.25,
    reference_node = 0)
with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
    o3d.pipelines.registration.global_optimization(
        POSE_GRAPH,
        o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
        o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(),
        options)

points_registered = []
for frame in FRAMES.values():
    if frame.frame_idx not in frame_map: raise Exception("FUCK")
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(frame.points)
    pcd.transform(POSE_GRAPH.nodes[frame_map[frame.frame_idx]].pose)
    points_registered.append(pcd)

o3d.visualization.draw_geometries(points_registered)