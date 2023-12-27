import json
import csv
import os

csv_file_path = os.path.abspath("CC-CVRP/csv_files/ccp_output.csv")
os.makedirs(os.path.dirname(csv_file_path), exist_ok=True)

ccp_output = open(csv_file_path,)

input = json.load(ccp_output)

# print(input)

cluster_file_path = os.path.abspath("CC-CVRP/csv_files/cluster_centroid_map.csv")
os.makedirs(os.path.dirname(cluster_file_path), exist_ok=True)

with open(cluster_file_path) as f:
    reader = csv.reader(f)
    centroid_paths = list(line for line in reader)
print(centroid_paths)

cluster_nodes = []
for path in centroid_paths:
    temp = [0]
    path.remove('0')
    for p in path:
        temp += input[str(int(p)-1)].get("nodes")
    cluster_nodes.append(temp)

print(cluster_nodes)

tsp_file_path = os.path.abspath("CC-CVRP/csv_files/tsp_input.csv")
os.makedirs(os.path.dirname(tsp_file_path), exist_ok=True)

with open(tsp_file_path, 'w', newline='') as file:
    writer = csv.writer(file)
    for cl in cluster_nodes:
        writer.writerow(cl)