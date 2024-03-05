import matplotlib.pyplot as plt
import numpy as np
import random
# from copy import deepcopy
import csv 
from recalculate import rec_edge, rec_intra_node

random.seed(0)

def plotMap(nodes, edges=[], colors = False):
    fig=plt.figure(figsize=(6,6), dpi= 100, facecolor='w', edgecolor='k')
    
    for e in (edges):
        start, end = e
        X = [nodes[start][0], nodes[end][0]]
        
        Y = [nodes[start][1], nodes[end][1]]
        plt.plot(X, Y, lw=1, ls="-", marker="", color = 'gray' )
        
    X = [c[0] for c in nodes]
    Y = [c[1] for c in nodes]
    S = [-c[2] for c in nodes]
    
    # plt.scatter(X, Y, S, c = S, cmap="RdYlGn", )
    # plt.scatter(X, Y, c = S, cmap="RdYlGn", )
    plt.scatter(X, Y, c = S, cmap="Greys" )
    for i in range(len(nodes)):
        plt.annotate(i, (X[i], Y[i]+0.2))
    plt.show()

# def get_min_value(array):
#     return np.min(array)

def get_min_index(array):
    min_index = np.argmin(array)
    min  = array[min_index]
    return min, min_index

def Euclidian_distance(coor_1, coor_2):
    x1, y1 = coor_1
    x2, y2 = coor_2
    return int(((x2 -x1)**2 + (y2-y1)**2 )**(1/2))

def create_dist_matrix_and_cost(data):
    NR_NODES = len(data)
    node_distances = np.zeros((NR_NODES,NR_NODES))
    cost_list = np.zeros(NR_NODES)
    for i in range(NR_NODES):
        for j in range(i,NR_NODES):
            dist = Euclidian_distance(data[i][:2],data[j][:2])

            cost_list[i] = data[i][2]
            node_distances[i][j] = dist
            node_distances[j][i] = dist
            # node_distances[i][i] = np.inf
    node_distances = node_distances
    return node_distances, cost_list

def create_cur_tour_from_list(lista, node_distances, cost_list):
    lenght = len(lista)
    total_cost = 0
    edges = []
    for i in range(lenght):
        a = lista[i]
        b = lista[(i+1)%lenght]
        dist = node_distances[a][b]
        cost = cost_list[i]
        total_cost += dist + cost 
        edges.append([a,b])
    return edges

def new_get_min_index(array):
    return np.argmin(array)

def get_data(path):
    with open(path, newline='') as csvfile:
        data = list(csv.reader(csvfile, delimiter=';'))
        for item in range(len(data)):
            i = data[item]
            data[item] = [int(x) for x in i]
    return data


def inter_swap_indexes(index_1, index_2, lista):
    node_1 = lista[index_1]
    lista[index_1] = lista[index_2]
    lista[index_2] = node_1
    return lista

def intra_swap_nodes(visited_node_index, unvisited_node, lista, unvisited):
    unvisited.remove(unvisited_node)
    unvisited.append(lista[visited_node_index])
    lista[visited_node_index] = unvisited_node
    return lista, unvisited

    
def check_total(lista, distance_matrix, cost_list = False):
    lenght = len(lista)
    total_cost = 0
    for i in range(lenght):
        a = lista[i]
        b = lista[(i+1)%lenght]
        dist = distance_matrix[a][b]
        if cost_list is not False:
            cost = cost_list[a]
            total_cost += cost 
        total_cost += dist
    return total_cost


def find_best(lista, total_cost, unvisited, distance_matrix, exchange = 'intra', cost_list = [], candidates = []):
    n = len(lista)

    lista_indexes = np.full((2*n),-1)
    for i in range(n):
        lista_indexes[lista[i]] = i
    
    delta_matrix = np.zeros((2*n,2*n))

    for visited_node_index in range(n):
        is_all_candidates_outside_lista = True
        for nr_candidate, candidate_node in enumerate(candidates[lista[visited_node_index]]):
            #only already nodes outside the cycle
            if lista_indexes[candidate_node] != -1:
                continue
            #if we check at least 10 candidates checkad and at least one was outside the lista to swap
            if nr_candidate >10 and not is_all_candidates_outside_lista:
                break
            is_all_candidates_outside_lista = False

            delta = rec_intra_node(visited_node_index, candidate_node, lista, distance_matrix, cost_list = cost_list)
            delta_matrix[lista[visited_node_index]][candidate_node] = delta
    
    best1, best2=np.unravel_index(np.argmin(delta_matrix, axis=None), delta_matrix.shape)
    best_score = delta_matrix[best1][best2]
    if best_score < 0:
        best1_index = lista_indexes[best1]
        new_lista, new_unvisited = intra_swap_nodes(best1_index, best2, lista, unvisited)
        return new_lista, new_unvisited, best_score + total_cost, False
    else:
        return lista, unvisited, total_cost, True

def edge_swap_indices(lista, i, j):
    new_lista = lista[:i+1]
    new_lista.append(lista[j])
    iter = j
    while True:
        iter -=1
        if iter == i:
            break
        new_lista.append(lista[iter])
        assert iter >= 0, f"{i, j, iter}"

    new_lista  += lista[j+1:]
    return new_lista


def find_best_edges(lista, total_cost, unvisited, distance_matrix, cost_list, candidates = []):
    n = len(lista)

    delta_matrix = np.zeros((2*n,2*n))
    lista_indexes = np.full((2*n),-1)
    for i in range(n):
        lista_indexes[lista[i]] = i
    
    for first_ind in range(n):
        is_all_candidates_in_lista = True
        #second_ind is the lista index of the candidate node
        for nr_candidate, second_ind in enumerate(lista_indexes[candidates[lista[first_ind]]]):
            #only already added edges to the lista should be considered
            if second_ind == -1:
                continue

            #if we check at least 10 candidates checkad and at least one was in the lista to swap
            if nr_candidate >10 and not is_all_candidates_in_lista:
                break

            is_all_candidates_in_lista = False


            #first index should be first in lista
            if first_ind > second_ind:
                _ = second_ind
                true_second_ind = first_ind
                true_first_ind = _
            else:
                true_second_ind = second_ind
                true_first_ind = first_ind
            #they cannot be consecutive nodes in lista
            diff = abs(true_second_ind - true_first_ind)
            if  diff == 1 or diff == n-1:
                continue
            delta = rec_edge(true_first_ind, true_second_ind, lista, distance_matrix)
            delta_matrix[lista[true_first_ind]][lista[true_second_ind]] = delta

    best1, best2=np.unravel_index(np.argmin(delta_matrix), delta_matrix.shape)
    best_score = delta_matrix[best1][best2]
    if best_score < 0:
        best1_index = lista_indexes[best1]
        best2_index = lista_indexes[best2]
        lista = edge_swap_indices(lista, best1_index, best2_index)
        return lista, unvisited, total_cost+best_score, False
    else: 
        return lista, unvisited, total_cost+best_score, True