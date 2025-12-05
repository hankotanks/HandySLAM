import os
import sys
import numpy as np
import open3d as o3d
import scipy
import cv2
import tqdm
import pickle
import copy
import glob

# discard all points with confidence < this value
CONFIDENCE_THRESHOLD = 2
# size of voxels to use in registration
VOXEL_SIZE = 0.04 # 2cm
# visualize the full pose graph instead of performing registration
VISUALIZE_GRAPH = False
# unload point clouds before running multiway registration
REDUCE_MEMORY_USAGE = True 
# number of points to write before splitting the cloud
POINTS_PER_CLOUD = 5_000_000

if VISUALIZE_GRAPH:
    import networkx as nx
    import matplotlib.pyplot as plt

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
        if not ret: 
            raise IOError(f"Could not read color frame {self.frame_idx} of {path_color}")
        frame_color = cv2.cvtColor(frame_color, cv2.COLOR_BGR2RGB).astype(np.float32)
        # load depthmap
        path_depth_frame = os.path.join(path_depth, f"{self.frame_idx:06d}.png")
        frame_depth = cv2.imread(path_depth_frame, cv2.IMREAD_UNCHANGED)
        if frame_depth is None: 
            raise IOError(f"Could not read depthmap {self.frame_idx} of {path_depth_frame}")
        frame_depth = frame_depth.astype(np.float32) / 1000.0;
        # load confidence map
        path_confidence_frame = os.path.join(path_confidence, f"{self.frame_idx:06d}.png")
        frame_confidence = cv2.imread(path_confidence_frame, cv2.IMREAD_UNCHANGED)
        if frame_confidence is None: 
            raise IOError(f"Could not read depthmap {self.frame_idx} of {path_confidence_frame}")
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
        x = (u - frame_depth_cx) * z / frame_depth_fx
        y = (v - frame_depth_cy) * z / frame_depth_fy
        self.points = np.transpose(np.vstack((x, y, z)))
        # store colors
        u_c = np.round(u * CAP_W / frame_depth_cols).astype(np.int32);
        v_c = np.round(v * CAP_H / frame_depth_rows).astype(np.int32);
        self.colors = frame_color[v_c, u_c] / 255.0

    def get_transformed_points(self, pose):
        ones = np.ones((self.points.shape[0], 1), dtype = self.points.dtype)
        points_homo = np.hstack([self.points, ones])
        return (points_homo @ pose.T)[:, 0:3]

path_blob = os.path.join(path_scene, "frames.pkl")

FRAMES = None
if(os.path.exists(path_blob)):
    print(f"Loading {os.path.basename(path_blob)}")
    with open(path_blob, "rb") as f:
        FRAMES = pickle.load(f)
else:
    FRAMES = {}
    with open(path_traj, "r") as f:
        lines = f.readlines()
        for line in tqdm.tqdm(lines, f"Building {os.path.basename(path_blob)}"):
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

EDGES_CLOSURES = np.argmin(np.abs(edges[:, :, None] - STAMPS[:, 0][None, None, :]), axis = 2)

EDGES_ADJACENT = np.array([[i, i+1] for i in range(EDGES_CLOSURES.min(), max(FRAMES.keys()))])
EDGES_CLOSURES = EDGES_CLOSURES[np.abs(EDGES_CLOSURES[:, 0] - EDGES_CLOSURES[:, 1]) != 1]

def calculate_robust_kernel_scale():
    residuals = []
    for edge in EDGES_ADJACENT:
        f0 = FRAMES[edge[0]]
        f1 = FRAMES[edge[1]]
        T_rel_slam = np.linalg.inv(f0.pose) @ f1.pose
        translation_residual = np.linalg.norm(T_rel_slam[:3, 3])
        R = T_rel_slam[:3, :3]
        angle_residual = np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))
        residuals.append((translation_residual, angle_residual))

    translations, rotations = zip(*residuals)
    translations = np.array(translations)
    rotations = np.array(rotations)
    scale = np.percentile(translations, 90)

    print(f"Robust kernel scale: {scale}")

    return scale

if VISUALIZE_GRAPH:
    print("Clustering edge graph")
    G = nx.Graph()
    for frame_idx in FRAMES.keys(): G.add_node(frame_idx)
    for edge in EDGES_ADJACENT: G.add_edge(edge[0], edge[1])
    for edge in EDGES_CLOSURES: G.add_edge(edge[0], edge[1])
    G_pos = nx.spring_layout(G, seed = 42, k = 0.5, iterations = 200, scale = 2.0)
    nx.draw_networkx_nodes(G, G_pos, cmap = plt.cm.tab20, node_size = 20)
    nx.draw_networkx_edges(G, G_pos, alpha = 0.3)
    plt.axis('off')
    plt.show()
    exit(0)

MAX_CORRESPONDENCE_DISTANCE = VOXEL_SIZE * 1.5

POSE_GRAPH = o3d.pipelines.registration.PoseGraph()

FRAME_MAP = {}
frame_count = 0
for frame_idx, frame in FRAMES.items():
    POSE_GRAPH.nodes.append(o3d.pipelines.registration.PoseGraphNode(np.linalg.inv(frame.pose)))
    FRAME_MAP[frame_idx] = frame_count
    frame_count += 1

def append_edges(edges, uncertain):
    for edge in edges:
        frame_fst = FRAMES[edge[0]]
        frame_snd = FRAMES[edge[1]]
        frame_fst_idx = FRAME_MAP[frame_fst.frame_idx]
        frame_snd_idx = FRAME_MAP[frame_snd.frame_idx]
        POSE_GRAPH.edges.append(o3d.pipelines.registration.PoseGraphEdge(
            frame_fst_idx, frame_snd_idx, 
            np.linalg.inv(frame_snd.pose) @ frame_fst.pose, 
            np.identity(6), uncertain = uncertain))

append_edges(EDGES_ADJACENT, False)
append_edges(EDGES_CLOSURES, True)

if REDUCE_MEMORY_USAGE:
    print(f"Unloading {os.path.basename(path_blob)} to reduce memory usage")
    for frame in FRAMES.values():
        frame.points = None
        frame.colors = None

options = o3d.pipelines.registration.GlobalOptimizationOption(
    max_correspondence_distance = MAX_CORRESPONDENCE_DISTANCE,
    edge_prune_threshold = calculate_robust_kernel_scale(),
    reference_node = 0,
    preference_loop_closure = 2.0)
with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
    o3d.pipelines.registration.global_optimization(
        POSE_GRAPH,
        o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
        o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(),
        options)

POSES = {}
for frame in FRAMES.values():
    if frame.frame_idx not in FRAME_MAP: 
        raise Exception(f"Frame {frame.frame_idx} had no equivalent node in pose graph")
    pose = POSE_GRAPH.nodes[FRAME_MAP[frame.frame_idx]].pose
    POSES[frame.frame_idx] = copy.deepcopy(pose)

if REDUCE_MEMORY_USAGE:
    print(f"Unloading pose graph")
    POSE_GRAPH = None
    print(f"Loading {os.path.basename(path_blob)}")
    with open(path_blob, "rb") as f: FRAMES = pickle.load(f)

class Output:
    _id = 0
    def __init__(self):
        self.id = type(self)._id
        type(self)._id += 1
        self.indices = []
        self.entries = 0
        self.positions = []
        self.colors = []
        self.timestamps = []
        self.labels = []

    def append(self, frame_idx):
        self.indices.append(frame_idx)
        frame_entries = FRAMES[frame_idx].points.shape[0]
        self.entries += frame_entries

        self.positions.append(FRAMES[frame_idx].get_transformed_points(POSES[FRAMES[frame_idx].frame_idx]))
        self.colors.append(FRAMES[frame_idx].colors)
        self.timestamps.append(np.repeat(FRAMES[frame_idx].t, frame_entries)[:, np.newaxis])
        self.labels.append(np.repeat(FRAMES[frame_idx].frame_idx, frame_entries)[:, np.newaxis].astype(np.int32))

        if self.entries < POINTS_PER_CLOUD: return False
        
        path_out = os.path.join(path_scene, f"out_{self.id:03}.ply")
        print(f"Writing {len(self.indices)} frames to {os.path.basename(path_out)}")

        device = o3d.core.Device("CPU:0")

        pcd = o3d.t.geometry.PointCloud(device)
        pcd.point.positions = o3d.core.Tensor(np.vstack(self.positions), o3d.core.float32, device)
        pcd.point.colors = o3d.core.Tensor(np.vstack(self.colors), o3d.core.float32, device)
        pcd.point.timestamps = o3d.core.Tensor(np.vstack(self.timestamps), o3d.core.float32, device)
        pcd.point.labels = o3d.core.Tensor(np.vstack(self.labels), o3d.core.int32, device)

        o3d.t.io.write_point_cloud(path_out, pcd)

        if REDUCE_MEMORY_USAGE:
            for frame_idx in self.indices:
                FRAMES[frame_idx].points = None
                FRAMES[frame_idx].colors = None
        
        return True

print("Removing old scene reconstruction")
for path_out_old in glob.glob(os.path.join(path_scene, "out_[0-9][0-9][0-9].ply")):
    os.remove(path_out_old)

out = Output()
for frame_idx in tqdm.tqdm(FRAMES.keys(), f"Building {os.path.basename(path_blob)}"):
    if out.append(frame_idx): out = Output()
