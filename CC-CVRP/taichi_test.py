import math
import numpy as np
import taichi as ti
import skfuzzy as fuzz
import vrplib as cvrplib

# Initialize Taichi with GPU support
ti.init(arch=ti.gpu)

instance = cvrplib.download_instance("A-n80-k10", "A-n80-k10.vrp")
instance = cvrplib.read_instance("A-n80-k10.vrp")

# 1. CALCULATE DISTANCES
# Access the edge weights from the dictionary
precalculated_distances = instance['edge_weight']
distances = [distance for distance in precalculated_distances]

# 2. GET NO. OF TRUCKS
trucks = instance['name'].partition("k")[2]
k = int(trucks)

# 3. GET CAPACITY OF TRUCKS
# Access the capacity from the dictionary
capacity = instance['capacity']

# 4. GET THE CUSTOMER COORDINATES
given_coords = instance['node_coord']
coords_original = np.array([coordinates for coordinates in given_coords], dtype=np.float32)

# Pairing up elements from flat_coords
paired_coords = coords_original.reshape((-1, 2))

# 5. GET DEPOT COORDINATES
depot = np.array(paired_coords[0])

# 6. GET DEMAND
given_demands = instance['demand']
demands = np.array([demand for demand in given_demands], dtype=np.float32)

# Constants
N = len(coords_original)
K = max(range(k, 2 * k + 1))

# Taichi fields
coords_field = ti.Vector.field(2, dtype=ti.f32, shape=N)
cntr_field = ti.Vector.field(2, dtype=ti.f32, shape=K)
u_field = ti.field(ti.f32, shape=(K, N))
cluster_membership_field = ti.field(ti.i32, shape=N)
WSS_field = ti.field(ti.f32, shape=(len(range(k, 2 * k + 1)),))

print(coords_field)

# # Copy data to Taichi fields
# @ti.kernel
# def copy_data():
#     for i in range(N):
#         coords_field[i] = coords_original[i]

#     for k in range(K):
#         cntr_field[k] = cntr[k]

#     for i in range(N):
#         cluster_membership_field[i] = cluster_membership[i]

# # Distance computation kernel
# @ti.kernel
# def compute_distances():
#     for i in range(N):
#         cluster_idx = cluster_membership_field[i]
#         dist = math.dist([coords_field[i][0], coords_field[i][1]],
#                          [cntr_field[cluster_idx][0], cntr_field[cluster_idx][1]])
#         WSS_field[0] += dist

# # Your existing code
# # ...

# # Your existing code related to fuzzy c-means clustering and WSS calculation
# WSS = []
# n_clusters_list = list(range(k, 2 * k + 1))

# for i in range(len(n_clusters_list)):
#     n_clusters = n_clusters_list[i]
#     coords = list(coords_original)

#     for c in range(n_clusters - 1):
#         coords.insert(0, depot)

#     coords = np.array(coords)

#     # Apply fuzzy c-means clustering
#     cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
#         coords.T, n_clusters, 2, error=0.005, maxiter=1000, init=None
#     )

#     # Predict cluster membership for each data point
#     cluster_membership = np.argmax(u, axis=0)

#     # Initialize WSS for the current number of clusters
#     WSS.append(0)

#     # Calculate WSS using Taichi for parallelized distance computations
#     copy_data()
#     compute_distances()

# # Retrieve the result from Taichi field
# WSS_result = WSS_field.to_numpy()[0]
# print(WSS_result)
