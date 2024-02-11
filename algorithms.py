import numpy as np
from utils import new_get_min_index

def greedy_cycle(distance_matrix,cost_list,NR_NODES, HALF_NODES, starting_node ):
    #pick second closest node
    modlify_cost = distance_matrix[starting_node]+cost_list[starting_node]
    modlify_cost[starting_node] = np.inf
    second_node = new_get_min_index(modlify_cost)

    non_visited = list(range(NR_NODES))
    non_visited.remove(starting_node)
    non_visited.remove(second_node)

    edges = [[starting_node, second_node],[second_node, starting_node]]
    cost = cost_list[starting_node] + cost_list[second_node] + 2 * distance_matrix[starting_node][second_node]

    for n in range(HALF_NODES-2):
        min_cost_diff = np.inf
        best_min_edge = [-1,-1]
        best_new_node = -1

        for e in edges:
            for new_node in non_visited:

                cost_diff = -distance_matrix[e[0]][e[1]] \
                    + distance_matrix[e[0]][ new_node] + distance_matrix[new_node][e[1]] \
                    + cost_list[new_node]

                if cost_diff < min_cost_diff:
                    best_min_edge = e
                    best_new_node = new_node
                    min_cost_diff = cost_diff

        cost += min_cost_diff
        non_visited.remove(best_new_node)

        edges.remove(best_min_edge)
        edges.append([best_min_edge[0], best_new_node])
        edges.append([best_new_node, best_min_edge[1]])
    return cost, edges