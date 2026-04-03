"""
Skeleton graph definition for ST-GCN using MediaPipe 33-joint pose landmarks.

Builds the adjacency matrix and spatial partitions that ST-GCN convolves over.
The graph topology encodes which joints are physically connected (bones).

Usage:
    graph = Graph()
    # graph.A is a (3, 33, 33) numpy array: the three spatial partitions
    # (self-connections, centripetal, centrifugal)
"""

import numpy as np

NUM_JOINTS = 33

# MediaPipe Pose landmark indices:
#  0: nose               11: left_shoulder     23: left_hip
#  1: left_eye_inner     12: right_shoulder    24: right_hip
#  2: left_eye           13: left_elbow        25: left_knee
#  3: left_eye_outer     14: right_elbow       26: right_knee
#  4: right_eye_inner    15: left_wrist        27: left_ankle
#  5: right_eye          16: right_wrist       28: right_ankle
#  6: right_eye_outer    17: left_pinky        29: left_heel
#  7: left_ear           18: right_pinky       30: right_heel
#  8: right_ear          19: left_index        31: left_foot_index
#  9: mouth_left         20: right_index       32: right_foot_index
# 10: mouth_right        21: left_thumb
#                        22: right_thumb

# Anatomical bone connections in the MediaPipe skeleton.
# Each tuple is (parent, child) following the natural kinematic chain.
MEDIAPIPE_EDGES = [
    # Face
    (0, 1), (1, 2), (2, 3),        # nose -> left eye chain
    (0, 4), (4, 5), (5, 6),        # nose -> right eye chain
    (3, 7),                         # left eye outer -> left ear
    (6, 8),                         # right eye outer -> right ear
    (9, 10),                        # mouth left -> mouth right
    (0, 9), (0, 10),               # nose -> mouth corners

    # Torso
    (11, 12),                       # left shoulder <-> right shoulder
    (11, 23),                       # left shoulder -> left hip
    (12, 24),                       # right shoulder -> right hip
    (23, 24),                       # left hip <-> right hip

    # Left arm
    (11, 13),                       # left shoulder -> left elbow
    (13, 15),                       # left elbow -> left wrist
    (15, 17),                       # left wrist -> left pinky
    (15, 19),                       # left wrist -> left index
    (15, 21),                       # left wrist -> left thumb

    # Right arm
    (12, 14),                       # right shoulder -> right elbow
    (14, 16),                       # right elbow -> right wrist
    (16, 18),                       # right wrist -> right pinky
    (16, 20),                       # right wrist -> right index
    (16, 22),                       # right wrist -> right thumb

    # Left leg
    (23, 25),                       # left hip -> left knee
    (25, 27),                       # left knee -> left ankle
    (27, 29),                       # left ankle -> left heel
    (27, 31),                       # left ankle -> left foot index

    # Right leg
    (24, 26),                       # right hip -> right knee
    (26, 28),                       # right knee -> right ankle
    (28, 30),                       # right ankle -> right heel
    (28, 32),                       # right ankle -> right foot index

    # Nose to shoulders (connects face to torso)
    (0, 11), (0, 12),
]

# Center of the skeleton for spatial partitioning.
# We use the midpoint concept: node 0 (nose) as root, and distances are
# measured in hops from the root. In practice, choosing either hip (23 or 24)
# or nose (0) works. We use 0 (nose) since it's the natural root of
# MediaPipe's kinematic tree.
CENTER_NODE = 0


def _build_hop_distance(num_joints: int, edges: list[tuple[int, int]]) -> np.ndarray:
    """Build shortest-path hop distance matrix using BFS from each node."""
    adj = np.zeros((num_joints, num_joints), dtype=int)
    for i, j in edges:
        adj[i, j] = 1
        adj[j, i] = 1

    dist = np.full((num_joints, num_joints), np.inf)
    for src in range(num_joints):
        dist[src, src] = 0
        visited = {src}
        queue = [src]
        while queue:
            node = queue.pop(0)
            for neighbor in range(num_joints):
                if adj[node, neighbor] and neighbor not in visited:
                    dist[src, neighbor] = dist[src, node] + 1
                    visited.add(neighbor)
                    queue.append(neighbor)
    return dist


def build_adjacency_matrix(
    num_joints: int = NUM_JOINTS,
    edges: list[tuple[int, int]] = MEDIAPIPE_EDGES,
) -> np.ndarray:
    """Build a symmetric binary adjacency matrix with self-loops.

    Returns:
        (num_joints, num_joints) float32 array. A[i,j] = 1 if joints i and j
        are connected or i == j.
    """
    A = np.eye(num_joints, dtype=np.float32)
    for i, j in edges:
        A[i, j] = 1.0
        A[j, i] = 1.0
    return A


def spatial_partition(
    num_joints: int = NUM_JOINTS,
    edges: list[tuple[int, int]] = MEDIAPIPE_EDGES,
    center: int = CENTER_NODE,
) -> np.ndarray:
    """Partition the adjacency into 3 subsets per the ST-GCN paper.

    For each edge (i, j) in the skeleton:
      - If i == j: belongs to partition 0 (self-loop / identity)
      - If j is closer to center than i: partition 1 (centripetal, toward root)
      - If j is farther from center than i: partition 2 (centrifugal, away from root)
      - If j is same distance as i: partition 1 (centripetal by convention)

    Returns:
        (3, num_joints, num_joints) float32 array of the three normalized
        adjacency partitions.
    """
    A = build_adjacency_matrix(num_joints, edges)
    hop_dist = _build_hop_distance(num_joints, edges)
    dist_to_center = hop_dist[center]

    # Partition 0: self-connections (identity)
    A_self = np.eye(num_joints, dtype=np.float32)

    # Partition 1: centripetal (neighbor closer to or same distance from center)
    A_close = np.zeros((num_joints, num_joints), dtype=np.float32)

    # Partition 2: centrifugal (neighbor farther from center)
    A_far = np.zeros((num_joints, num_joints), dtype=np.float32)

    for i in range(num_joints):
        for j in range(num_joints):
            if A[i, j] == 0 or i == j:
                continue
            if dist_to_center[j] <= dist_to_center[i]:
                A_close[i, j] = 1.0
            else:
                A_far[i, j] = 1.0

    # Normalize each partition: D^(-1/2) A D^(-1/2)
    partitions = np.stack([A_self, A_close, A_far])  # (3, V, V)
    for k in range(3):
        partitions[k] = _normalize(partitions[k])

    return partitions


def _normalize(A: np.ndarray) -> np.ndarray:
    """Symmetric normalization: D^(-1/2) A D^(-1/2).

    Handles zero-degree nodes by leaving their rows/cols as zero.
    """
    D = np.sum(A, axis=1)
    D_inv_sqrt = np.zeros_like(D)
    nonzero = D > 0
    D_inv_sqrt[nonzero] = 1.0 / np.sqrt(D[nonzero])
    D_inv_sqrt = np.diag(D_inv_sqrt)
    return D_inv_sqrt @ A @ D_inv_sqrt


class Graph:
    """MediaPipe 33-joint skeleton graph for ST-GCN.

    Attributes:
        A: (3, 33, 33) float32 ndarray -- the three normalized spatial
           partitions (self, centripetal, centrifugal).
        num_joints: 33
        edges: list of (src, dst) bone connections
    """

    def __init__(self):
        self.num_joints = NUM_JOINTS
        self.edges = MEDIAPIPE_EDGES
        self.A = spatial_partition(NUM_JOINTS, MEDIAPIPE_EDGES, CENTER_NODE)

    def __repr__(self) -> str:
        return (
            f"Graph(num_joints={self.num_joints}, "
            f"num_edges={len(self.edges)}, "
            f"partitions={self.A.shape})"
        )
