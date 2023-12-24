"""LINE 218 and 210 latest problems
"""

# import os
# import vrplib

# # Set the environment variable
# os.environ['DATASET'] = 'X-n1001-k43.vrp'

# # Retrieve the value of the environment variable and assign it to dataset
# dataset = os.getenv('DATASET')

# # Check if the assignment was successful
# if dataset is not None:
#     print(f'Dataset path: {dataset}')
# else:
#     print('The environment variable DATASET is not set.')

# vrplib.download_instance("X-n1001-k43", "X-n1001-k43.vrp")
# instance = vrplib.read_instance("X-n1001-k43.vrp")
# print(instance)
# vrplib.list_names()

# import vrplib as cvrplib

# instance = cvrplib.download_instance("X-n1001-k43","A\~$n32-k5.vrp")

# print(instance.__dict__)

# help(cvrplib)

###############################################################################################################################


import math
import vrplib as cvrplib

instance=cvrplib.download_instance("X-n1001-k43", "X-n1001-k43.vrp")
instance = cvrplib.read_instance("X-n1001-k43.vrp")

print(instance)

# # 1. CALCULATE DISTANCES
# # Access the edge weights from the dictionary
# precalculated_distances = instance['edge_weight']
# distances = [distance for distance in precalculated_distances]

# # Save in flat_distances and print           
# flat_distances = [item for sublist in distances for item in sublist]

# print(distances)

# # 2. GET NO. OF TRUCKS
# trucks = instance['name'].partition("k")[2]
# k=int(trucks)
# print(k)

# # 3. GET CAPACITY OF TRUCKS
# # Access the capacity from the dictionary
capacity = instance['capacity']

# # Print the capacity
print(capacity)


# # 4. GET THE CUSTOMER COORDINATES
# given_coords=instance['node_coord']
# coords_original = [coordinates for coordinates in given_coords]

# flat_coords = [item for sublist in coords_original for item in sublist]

# # Pairing up elements from flat_coords
# paired_coords = [(flat_coords[i], flat_coords[i + 1]) for i in range(0, len(flat_coords), 2)]

# print(paired_coords)



# # 5. GET DEPOT COORDINATES
# depot = paired_coords[0]
# print(depot)


# # 6. GET DEMAND
# given_demands=instance['demand']
# demands = [int(demand) for demand in given_demands]

# print(demands)



# import json
# import vrplib as cvrplib
# from ccp_input import ret_instance

# ccp_output = open("./csv_files/ccp_output.csv",)

# input = json.load(ccp_output)

# instance=cvrplib.download_instance("X-n1001-k43", "X-n1001-k43.vrp")
# d_instance = cvrplib.read_instance("X-n1001-k43.vrp")

# print(d_instance['name'])

# instance = ret_instance(input, d_instance)

# n=instance.dimension
# print(n)
# strp=instance.name.partition("k")[2]
# p=int(strp)#Number of trucks
# print(p)
# distances=instance.distances
# print(distances)
# node_list = list(range(1, n))
# print(node_list)



# import csv
# import numpy as np

# csv_filename = "./csv_files/tsp_input.csv"
# with open(csv_filename) as f:
#     reader = csv.reader(f)
#     centroid_paths = list(line for line in reader)
# node_list = [list(map(int, lst)) for lst in centroid_paths]


# max_length = max(len(node) for node in node_list)

# # Pad the shorter lists with zeros to make them the same length
# padded_node_list = [node + [0] * (max_length - len(node)) for node in node_list]

# # Convert the padded_node_list to a NumPy array
# f_node_list = np.array(padded_node_list)


# # f_node_list = np.array(node_list) #Used to convert the node_list into a numpy array

# print(f_node_list)


# import math
# import vrplib as cvrplib
# import csv
# import numpy as np

# #DISTANCES
# instance=cvrplib.download_instance("X-n1001-k43", "X-n1001-k43.vrp")
# instance = cvrplib.read_instance("X-n1001-k43.vrp")

# precalculated_distances = instance['edge_weight']
# distances = [distance for distance in precalculated_distances]

# #Save in flat_distances and print           
# distances = [item for sublist in distances for item in sublist]

# import csv
# import numpy as np

# # #NODE_LIST
# csv_filename = "./csv_files/tsp_input.csv"
# with open(csv_filename) as f:
#     reader = csv.reader(f)
#     centroid_paths = list(line for line in reader)
# node_list = [list(map(int, lst)) for lst in centroid_paths]

# max_length = max(len(node) for node in node_list)

# # Pad the shorter lists with zeros to make them the same length
# padded_node_list = [node + [0] * (max_length - len(node)) for node in node_list]

# # Convert the padded_node_list to a NumPy array
# node_list = np.array(padded_node_list)

# print(node_list)
# # Find the maximum length of inner lists
# max_length = max(len(node) for node in node_list)

# # Pad the shorter lists with zeros to make them the same length
# padded_node_list = [node + [0] * (max_length - len(node)) for node in node_list]

# # Convert the padded_node_list to a NumPy array
# node_list = np.array(padded_node_list)


# n = len(node_list[0])

# # # Reshape distances into a 2D array
# # distances = np.array(distances).reshape((n, n))


# if len(distances) == 64:
#     reshaped_arr = distances.reshape(8, 8)
# else:
#     print("Cannot reshape the array into (8, 8) because the total number of elements is not 64.")


# import vrplib as cvrplib

# solution=cvrplib.download_solution("X-n1001-k43", "X-n1001-k43.sol")
# solution = cvrplib.read_solution("X-n1001-k43.sol")

# print("\nClassical Solution Route\t",solution['routes'])
# print("\nClassical Solution Total cost:\t",solution['cost'])

# TSP INPUT CSV
# 0,7,13,19,31,17,21
# 0,3,2,23,4,11,28,8
# 0,16,12,30,26,1
# 0,24,27,14,20
# 0,22,9,18,6,10,25,5,29,15


