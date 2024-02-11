import random

def calculate_total_distance(order, distance_matrix):
    total_distance = 0
    num_nodes = len(order)

    for i in range(num_nodes - 1):
        total_distance += distance_matrix[order[i]][order[i + 1]]

    # Return to the starting node
    total_distance += distance_matrix[order[-1]][order[0]]

    return total_distance

def calculate_delta_distance(order, distance_matrix, node1, node2):
    num_nodes = len(order)

    # Calculate the original distances
    original_distance = (
        distance_matrix[order[node1 - 1]][order[node1 % num_nodes]] +
        distance_matrix[order[node1]][order[(node1 + 1) % num_nodes]] +
        distance_matrix[order[node2 - 1]][order[node2 % num_nodes]] +
        distance_matrix[order[node2]][order[(node2 + 1) % num_nodes]]
    )

    # Calculate the new distances after the exchange
    new_distance = (
        distance_matrix[order[node1 - 1]][order[node2 % num_nodes]] +
        distance_matrix[order[node2]][order[(node1 + 1) % num_nodes]] +
        distance_matrix[order[node2 - 1]][order[node1 % num_nodes]] +
        distance_matrix[order[node1]][order[(node2 + 1) % num_nodes]]
    )
    return new_distance - original_distance

def exchange_nodes(order, node1, node2):
    new_order = order.copy()
    new_order[node1], new_order[node2] = new_order[node2], new_order[node1]
    return new_order

def tsp_exchange_nodes_with_deltas(distance_matrix, iterations=1000):
    num_nodes = len(distance_matrix)
    current_order = list(range(num_nodes))
    current_distance = calculate_total_distance(current_order, distance_matrix)

    for _ in range(iterations):
        # Randomly select two distinct nodes to exchange
        node1, node2 = random.sample(range(num_nodes), 2)

        # Calculate the change in distance (delta)
        delta_distance = calculate_delta_distance(current_order, distance_matrix, node1, node2)

        # If the new order is better, update the current order and distance
        if delta_distance < 0:
            current_order = exchange_nodes(current_order, node1, node2)
            current_distance += delta_distance

    return current_order, current_distance

# Example usage with the provided distance matrix
distance_matrix = [
    [0., 100., 200., 300., 316., 223., 141., 100.],
    [100., 0., 100., 200., 223., 141., 100., 141.],
    [200., 100., 0., 100., 141., 100., 141., 223.],
    [300., 200., 100., 0., 100., 141., 223., 316.],
    [316., 223., 141., 100., 0., 100., 200., 300.],
    [223., 141., 100., 141., 100., 0., 100., 200.],
    [141., 100., 141., 223., 200., 100., 0., 100.],
    [100., 141., 223., 316., 300., 200., 100., 0.]
]

best_order, best_distance = tsp_exchange_nodes_with_deltas(distance_matrix)
print("Best Order:", best_order)
print("Best Distance:", best_distance)
