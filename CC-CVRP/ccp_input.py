import math
import taichi as ti

# Initialize Taichi
ti.init(arch=ti.cpu)  # Use ti.gpu for GPU-based computing if needed

# Define a Taichi kernel for distance calculation with type annotations
@ti.kernel
def calculate_distances(x: ti.template(), y: ti.template(), distances: ti.template()):
    n = x.shape[0]  
    for i in range(n):
        for j in range(n):
            abscissa = (x[i][0] - y[j][0])**2  
            ordinate = (x[i][1] - y[j][1])**2  
            distances[i, j] = ti.sqrt(abscissa + ordinate)

def find_dist_mat(x, y):
    n = len(x)
    distances = ti.field(ti.f32, shape=(n, n))
    x_data = ti.Vector.field(2, dtype=ti.f32, shape=n)
    y_data = ti.Vector.field(2, dtype=ti.f32, shape=n)

    for i in range(n):
        x_data[i] = x[i]
        y_data[i] = y[i]

    calculate_distances(x_data, y_data, distances)

    # Retrieve the distances as a NumPy array
    distances_np = distances.to_numpy()

    return distances_np

class objectview(object):
    def __init__(self, d):
        self.__dict__ = d

def get_coord(data, data_instance):
    given_coords = data_instance['node_coord']
    coords_original = [coordinates for coordinates in given_coords]
    flat_coords = [item for sublist in coords_original for item in sublist]
    coords = [(flat_coords[i], flat_coords[i + 1]) for i in range(0, len(flat_coords), 2)]

    coordinates = []
    coordinates.append(coords[0])
    for i in data.values():
        coordinates.append(i.get("centroid"))
    return coordinates

def get_demands(data):
    demands = [0]
    for i in data.values():
        demands.append(i.get('demand'))
    return demands

def ret_instance(input, data_instance):
    coordinates = get_coord(input, data_instance)
    distances = find_dist_mat(coordinates, coordinates)
    n = len(input) + 1

    instance = {
        "name": data_instance['name'],
        "dimension": n,
        "n_customers": len(input),
        "depot": 0,
        "customers": [i for i in range(1, n)],
        "capacity": data_instance['capacity'],
        "distances": distances.tolist(),
        "demands": get_demands(input),
        "coordinates": coordinates,
        "distance_limit": math.inf,
        "service_times": [0.0] * n
    }

    return objectview(instance)
