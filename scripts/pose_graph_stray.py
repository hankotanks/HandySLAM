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
VOXEL_SIZE = 0.02 # 2cm
# visualize the full pose graph instead of performing registration
VISUALIZE_GRAPH = False
# unload point clouds before running multiway registration
REDUCE_MEMORY_USAGE = True 
# number of points to write before splitting the cloud
# POINTS_PER_CLOUD = 10_000_000
# toggle multiway registration
OPTIMIZE_POSES = True
# maximum depth to use for registration (meters)
MAX_DEPTH = 3.0

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
        if frame_confidence.shape != frame_depth.shape:
            print("Resizing confidence frame")
            frame_confidence = cv2.resize(frame_confidence, frame_depth.shape, interpolation = cv2.INTER_LINEAR)
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
        z_filter = (z > 0.0) & (z < MAX_DEPTH) & (frame_confidence.flatten() >= CONFIDENCE_THRESHOLD)
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

# expects 0 timestamp1 timestamp2 or normal edges
# expects 1 timestamp1 timestamp2 for loop closure edges
edges_raw = np.loadtxt(path_edges, usecols = [0, 1, 2])
if edges_raw.shape[1] != 3: raise IOError(f"{path_edges} was malformed")

EDGES = np.argmin(np.abs(edges_raw[:, 1:][:, :, None] - STAMPS[:, 0][None, None, :]), axis = 2)
EDGES_CLOSURES = EDGES[edges_raw[:, 0].astype(np.int32) == 1]
EDGES_ADJACENT = EDGES[edges_raw[:, 0].astype(np.int32) == 0]
if(EDGES_ADJACENT.shape[0] + EDGES_CLOSURES.shape[0] != EDGES.shape[0]):
    raise IOError(f"{path_edges} was malformed")

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

def calcuate_radius(pcd):
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)
    pt = np.asarray(pcd.points)[0]
    _, idx, dist = pcd_tree.search_knn_vector_3d(pt, 2)
    return np.sqrt(dist[1])

MAX_CORRESPONDENCE_DISTANCE = VOXEL_SIZE * 1.5

if VISUALIZE_GRAPH:
    print("Clustering edge graph")
    G = nx.Graph()
    for frame_idx in FRAMES.keys(): 
        G.add_node(frame_idx, pos = FRAMES[frame_idx].pose[0:3, 3])
    for edge in EDGES_ADJACENT: G.add_edge(edge[0], edge[1])
    for edge in EDGES_CLOSURES: G.add_edge(edge[0], edge[1])
    

    count = 0
    final = None
    while len(list(G.nodes)) > 1:
        dist_min = float('inf')
        edge_min = None
        for u, v in G.edges:
            dist = np.linalg.norm(G.nodes[u]['pos'] - G.nodes[v]['pos'])
            if dist < dist_min:
                dist_min = dist
                edge_min = (u, v)

        frame_fst = FRAMES[edge_min[0]]
        frame_snd = FRAMES[edge_min[1]]
        pcd_fst = o3d.geometry.PointCloud()
        pcd_fst.points = o3d.utility.Vector3dVector(frame_fst.get_transformed_points(frame_fst.pose))
        pcd_fst.estimate_normals(search_param = o3d.geometry.KDTreeSearchParamHybrid(radius = calcuate_radius(pcd_fst), max_nn = 30))
        pcd_fst.orient_normals_consistent_tangent_plane(k = 20)
        pcd_snd = o3d.geometry.PointCloud()
        pcd_snd.points = o3d.utility.Vector3dVector(frame_snd.get_transformed_points(frame_snd.pose))
        pcd_snd.estimate_normals(search_param = o3d.geometry.KDTreeSearchParamHybrid(radius = calcuate_radius(pcd_snd), max_nn = 30))
        pcd_snd.orient_normals_consistent_tangent_plane(k = 20)
        result = o3d.pipelines.registration.registration_icp(
            pcd_fst, pcd_snd,
            max_correspondence_distance = MAX_CORRESPONDENCE_DISTANCE,
            estimation_method = o3d.pipelines.registration.TransformationEstimationPointToPlane())

        frame_fst.points = np.vstack([
            frame_fst.points, 
            frame_snd.get_transformed_points(np.linalg.inv(frame_fst.pose) @ result.transformation @ frame_snd.pose)])
        for neighbor in G.neighbors(edge_min[1]):
            if neighbor != edge_min[0]:
                G.add_edge(neighbor, edge_min[0])
        G.remove_node(edge_min[1])

        print(f"Merged {edge_min[1]} to {edge_min[0]}. Graph contains {len(list(G.nodes))} nodes")
        final = edge_min[0]

        count += 1
        if count % 3 == 0:
            largest = 0
            largest_idx = None
            for frame in FRAMES.values():
                if frame.points.shape[0] > largest:
                    largest = frame.points.shape[0]
                    largest_idx = frame.frame_idx

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(FRAMES[largest_idx].points)

            o3d.visualization.draw_geometries([pcd])

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(FRAMES[final].points)

    o3d.visualization.draw_geometries([pcd])
    
    # plt.show()
    exit(0)

POSES = {}
if OPTIMIZE_POSES:
    POSE_GRAPH = o3d.pipelines.registration.PoseGraph()

    FRAME_MAP = {}
    frame_count = 0
    for frame_idx, frame in FRAMES.items():
        # POSE_GRAPH.nodes.append(o3d.pipelines.registration.PoseGraphNode(np.eye(4)))
        POSE_GRAPH.nodes.append(o3d.pipelines.registration.PoseGraphNode(frame.pose))
        FRAME_MAP[frame_idx] = frame_count
        frame_count += 1

    def append_edges(edges, uncertain, pairwise):
        for edge in tqdm.tqdm(edges, f"Adding {'uncertain' if uncertain else 'certain'} edges"):
            frame_fst = FRAMES[edge[0]]
            frame_snd = FRAMES[edge[1]]
            frame_fst_idx = FRAME_MAP[frame_fst.frame_idx]
            frame_snd_idx = FRAME_MAP[frame_snd.frame_idx]
            edge_transformation = np.linalg.inv(frame_snd.pose) @ frame_fst.pose
            edge_information = np.identity(6)
            if pairwise:
                pcd_fst = o3d.geometry.PointCloud()
                pcd_fst.points = o3d.utility.Vector3dVector(frame_fst.points)
                pcd_fst.estimate_normals(search_param = o3d.geometry.KDTreeSearchParamHybrid(radius = calcuate_radius(pcd_fst), max_nn = 30))
                pcd_fst.orient_normals_consistent_tangent_plane(k = 20)
                # pcd_fst.points = o3d.utility.Vector3dVector(frame_fst.get_transformed_points(frame_fst.pose))
                pcd_snd = o3d.geometry.PointCloud()
                pcd_snd.points = o3d.utility.Vector3dVector(frame_snd.points)
                pcd_snd.estimate_normals(search_param = o3d.geometry.KDTreeSearchParamHybrid(radius = calcuate_radius(pcd_snd), max_nn = 30))
                pcd_snd.orient_normals_consistent_tangent_plane(k = 20)
                # pcd_snd.points = o3d.utility.Vector3dVector(frame_snd.get_transformed_points(frame_snd.pose))
                result = o3d.pipelines.registration.registration_icp(
                    pcd_fst, pcd_snd,
                    max_correspondence_distance = MAX_CORRESPONDENCE_DISTANCE,
                    init = np.linalg.inv(frame_snd.pose) @ frame_fst.pose,
                    estimation_method = o3d.pipelines.registration.TransformationEstimationPointToPlane())
                edge_transformation = result.transformation
                edge_information = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
                    pcd_fst, pcd_snd, 
                    max_correspondence_distance = MAX_CORRESPONDENCE_DISTANCE, 
                    transformation = edge_transformation)

            POSE_GRAPH.edges.append(o3d.pipelines.registration.PoseGraphEdge(
                frame_fst_idx, frame_snd_idx, 
                edge_transformation, 
                edge_information, uncertain = uncertain))

    append_edges(EDGES_ADJACENT, False, True)
    append_edges(EDGES_CLOSURES, True, False)

    if REDUCE_MEMORY_USAGE:
        print(f"Unloading {os.path.basename(path_blob)} to reduce memory usage")
        for frame in FRAMES.values():
            frame.points = None
            frame.colors = None

    options = o3d.pipelines.registration.GlobalOptimizationOption(
        max_correspondence_distance = MAX_CORRESPONDENCE_DISTANCE,
        edge_prune_threshold = calculate_robust_kernel_scale(),
        reference_node = 0)
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug):
        o3d.pipelines.registration.global_optimization(
            POSE_GRAPH,
            o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
            o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(),
            options)

    for frame in FRAMES.values():
        if frame.frame_idx not in FRAME_MAP: 
            raise Exception(f"Frame {frame.frame_idx} had no equivalent node in pose graph")
        POSES[frame.frame_idx] = copy.deepcopy(POSE_GRAPH.nodes[FRAME_MAP[frame.frame_idx]].pose)

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
        self.cloud = None

    def append(self, frame_idx):
        if self.cloud is not None:
            raise Exception("Cannot call Output.append after calling Output.build")

        self.indices.append(frame_idx)
        frame_entries = FRAMES[frame_idx].points.shape[0]
        self.entries += frame_entries

        self.positions.append(FRAMES[frame_idx].get_transformed_points(FRAMES[frame_idx].pose))
        self.colors.append(FRAMES[frame_idx].colors)
        self.timestamps.append(np.repeat(FRAMES[frame_idx].t, frame_entries)[:, np.newaxis])
        self.labels.append(np.repeat(FRAMES[frame_idx].frame_idx, frame_entries)[:, np.newaxis].astype(np.int32))

    def build(self):
        device = o3d.core.Device("CPU:0")

        self.cloud = o3d.t.geometry.PointCloud(device)
        self.cloud.point.positions = o3d.core.Tensor(np.vstack(self.positions), o3d.core.float32, device)
        self.cloud.point.colors = o3d.core.Tensor(np.vstack(self.colors), o3d.core.float32, device)
        self.cloud.point.timestamps = o3d.core.Tensor(np.vstack(self.timestamps), o3d.core.float32, device)
        self.cloud.point.labels = o3d.core.Tensor(np.vstack(self.labels), o3d.core.int32, device)

        if REDUCE_MEMORY_USAGE:
            for frame_idx in self.indices:
                FRAMES[frame_idx].points = None
                FRAMES[frame_idx].colors = None
    
    def write(self):
        path_out = os.path.join(path_scene, f"out_{self.id:03}.ply")
        o3d.t.io.write_point_cloud(path_out, self.cloud)

print("Removing old scene reconstruction")
for path_out_old in glob.glob(os.path.join(path_scene, "out_[0-9][0-9][0-9].ply")):
    os.remove(path_out_old)

out = Output()
for frame_idx in tqdm.tqdm(FRAMES.keys(), f"Writing Cloud"):
    out.append(frame_idx)

out.build()
out.write()
