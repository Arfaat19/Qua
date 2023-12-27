import numpy as np
import vrplib as cvrplib

from dimod import ConstrainedQuadraticModel, Binary, Integer
from dimod import quicksum
from dwave.system import LeapHybridCQMSampler

import time
import csv
import json

import taichi as ti

from ccp_input import ret_instance


# colors = [ "#58FF33", "#CD6155", "#DAF7A6", "#FFC300", "#A569BD", "#5499C7", "#45B39D", "#6E2C00", "#FF33D1", "#FFFFFF", "#000000", "#33FFAF", "#33FFE0", "#FF3333"]

ccp_output = open("CC-CVRP\csv_files\ccp_output.csv",)

input = json.load(ccp_output)


instance=cvrplib.download_instance("A-n32-k5", "A-n32-k5.vrp")
d_instance = cvrplib.read_instance("A-n32-k5.vrp")

instance = ret_instance(input, d_instance)

n=instance.dimension
strp=instance.name.partition("k")[2]
p=int(strp)#Number of trucks
distances=instance.distances
node_list = list(range(1, n))

c=np.zeros((n,n))
for i in range(0,n):
    for j in range(0,n):
        c[i][j]=distances[i][j]
        
V=int(n*(n-1)*p)
D=instance.demands
Q=instance.capacity

x = np.array([[[Binary(f'x_{r}_{i}_{j}') for j in range(n)] for i in range(n)] for r in range(p)])

cqm = ConstrainedQuadraticModel()

cqm.set_objective(quicksum(c[i][j]*x[r][i][j] for r in range(p) for i in range(n) for j in range(n) if(j!=i)))

# Each node is visited only once
for j in range(1,n):
    cqm.add_constraint(quicksum(x[r][i][j] for r in range(p) for i in range(n) if i!=j)==1, label=f'Constraint1_{j}')

# Each vehicle must leave the depot
for r in range(p):
    cqm.add_constraint(quicksum(x[r][0][j] for j in range(1,n))==1, label=f'Constraint2_{r}')

# The order of the route is valid and maintained
for j in range(n):
    for r in range(p):
        cqm.add_constraint(quicksum(x[r][i][j] for i in range(n) if i!=j)-quicksum(x[r][j][i] for i in range(n) if i!=j)==0,label=f'Constraint3_{j}_{r}')

B = n

s_max = n

s =[[None]*(p) for _ in range(n)]

for i in range(n):
    for r in range(p):
        s[i][r] = Integer(lower_bound=1,upper_bound = s_max,label=f't.{i}.{r}')

for i in range(1,n):
    for j in range(1,n):
        if i!=j:
            for r in range(p):
                cqm.add_constraint((s[j][r]-(s[i][r]+1)+B*(1-x[r][i][j]))>=0)

# Each vehicle does not exceed its capacity
for r in range(p):
    cqm.add_constraint(quicksum(D[j]*x[r][i][j] for i in range(n) for j in range(1,n) if j!=i)<=Q,label=f'Constraint4_{r}')



def get_token():
    dwave_token='DEV-206e895fcd45e66ad6802ef108a574189f3389fc'
    return dwave_token   



print("\nStarting D wave\n")
startime=time.time()
sampler=LeapHybridCQMSampler(token=get_token())
sampleset = sampler.sample_cqm(cqm,time_limit=150 ,label='CVRP')
feasible_sampleset=sampleset.filter(lambda row:row.is_feasible)
end_time=time.time()
try:
    best_solution=feasible_sampleset.first.sample
except:
    print("No feasible solution found")
    exit()

print("\n Total execution time for "+str(n)+" nodes "+str(p)+" vehicles "+"takes : "+str(round((end_time-startime),2))+" seconds\n")

truck_stops=[]
routes=[[] for _ in range(p)]
for key,val in best_solution.items():
    if val==1.0:
        if "x_" in key:
            truck_stops.append(key.split('_')[1:])
            routes[int(truck_stops[-1][0])].append(truck_stops[-1][1:])

paths = []

total_cost=0
for i in range(p):
    current_route=routes[i]
    current_cost=0
    temp = []
    for r in current_route:
        current_cost+=c[int(r[0])][int(r[1])]
    print("\nTruck",i,"route:")
    for r in current_route:
        print(r)
        temp+=r
    print("\nTruck",i,"cost:",round(current_cost,2))
    total_cost+=current_cost
    paths.append(list(set(temp)))

print(paths)

with open('CC-CVRP\csv_files\cluster_centroid_map.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    for cl in paths:
        writer.writerow(cl)

# time_limit = sampler.min_time_limit(cqm)
# print(time_limit)

# def min_time_limit(self, cqm: dimod.ConstrainedQuadraticModel) -> float:
#     """Return the minimum `time_limit` accepted for the given problem."""

#     # todo: remove the hard-coded defaults
#     num_variables_multiplier = self.properties.get('num_variables_multiplier', 1.57e-04)
#     num_biases_multiplier = self.properties.get('num_biases_multiplier', 4.65e-06)
#     num_constraints_multiplier = self.properties.get('num_constraints_multiplier', 6.44e-09)
#     minimum_time_limit = self.properties['minimum_time_limit_s']

#     num_variables = len(cqm.variables)
#     num_constraints = len(cqm.constraints)
#     num_biases = cqm.num_biases()

#     return max(
#         num_variables_multiplier * num_variables +
#         num_biases_multiplier * num_biases +
#         num_constraints_multiplier * num_variables * num_constraints,
#         minimum_time_limit
#     )
