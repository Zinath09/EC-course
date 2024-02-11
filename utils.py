import matplotlib.pyplot as plt
import numpy as np
import random
# from copy import deepcopy
import csv 
from recalculate import rec_inter_node, rec_edge, rec_intra_node

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
    S = [c[2] for c in nodes]
    
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


def inter_swap_nodes(index_1, index_2, lista):
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

def find_first_better(lista, total_cost, unvisited, distance_matrix, exchange = "intra", cost_list = []):
    random.shuffle(unvisited)
    
    random_lista_indexes = list(range(len(lista)))
    random.shuffle(random_lista_indexes)

    # n = len(random_lista)
    if exchange == 'inter':
        for first_ind in range(len(lista)):
            for second_ind in range(first_ind+1, len(lista)):
                new_total = total_cost + rec_inter_node(random_lista_indexes[first_ind], random_lista_indexes[second_ind], lista, distance_matrix)
                
                if new_total < total_cost:
                    new_lista = inter_swap_nodes(random_lista_indexes[first_ind], random_lista_indexes[second_ind], lista)

                    return new_lista, unvisited, new_total, False
                
    elif exchange == 'intra':
        for unvisited_node in unvisited:
            for visited_node_index in random_lista_indexes:

                # new_total = check_total(new_lista, distance_matrix, cost_list)
                new_total = total_cost + rec_intra_node(visited_node_index, unvisited_node, lista, distance_matrix, cost_list)
                

                if new_total < total_cost:
                    new_lista, new_unvisited = intra_swap_nodes(visited_node_index, unvisited_node, lista, unvisited)
                    return new_lista, new_unvisited, new_total, False
                
    return lista, unvisited, total_cost, True


def find_best(lista, total_cost, unvisited, distance_matrix, exchange = 'intra', cost_list = []):
    random_lista_indexes = list(range(len(lista)))
    
    random.shuffle(random_lista_indexes)
    best = -1, -1, total_cost, True

    if exchange == 'inter':
        for first_ind in range(len(lista)):
            for second_ind in range(first_ind+1, len(lista)):
                new_total = total_cost + rec_inter_node(random_lista_indexes[first_ind], random_lista_indexes[second_ind], lista, distance_matrix)
                if new_total < best[2]:
                    best = random_lista_indexes[first_ind], random_lista_indexes[second_ind], new_total, False

        first, second, new_total, terminate = best
        if terminate:
            return lista, unvisited, total_cost, terminate
        else:
            new_lista = inter_swap_nodes(first, second, lista)
            return new_lista, unvisited, new_total, terminate

    elif exchange == 'intra':
        for unvisited_node in unvisited:
            for visited_node_index in random_lista_indexes:
                new_total = total_cost + rec_intra_node(visited_node_index, unvisited_node, lista, distance_matrix, cost_list = cost_list)

                if new_total < best[2]:
                    best = visited_node_index, unvisited_node, new_total, False

        visited_node_index, unvisited_node, new_total, terminate = best

        if terminate:
            return lista, unvisited, total_cost, terminate
        else:
            new_lista, new_unvisited = intra_swap_nodes(visited_node_index, unvisited_node, lista, unvisited)
            return new_lista, new_unvisited, new_total, terminate


def edge_swap_indices(lista, i, j):

    new_lista = lista[:i+1]
    new_lista.append(lista[j])

    iter = j
    while True:
        iter -=1
        if iter == i:
            break
        new_lista.append(lista[iter])
        if iter <0:
            # break
            print(i, j, iter)
            assert(False)


    new_lista  += lista[j+1:]
    return new_lista



def find_first_better_edges(lista, total_cost, unvisited, distance_matrix, cost_list):
    random.shuffle(unvisited)
    n = len(lista)
    random_indexes = list(range(len(lista)))

    random.shuffle(random_indexes)

    
    for first_ind in range(len(lista)):
        for second_ind in range(first_ind+1, len(lista)):

            if abs(random_indexes[first_ind] - random_indexes[second_ind]) == 1:
                continue

            if random_indexes[first_ind] > random_indexes[second_ind]:
                _ = second_ind
                second_ind = first_ind
                first_ind = _
            
            # new_lista = deepcopy(lista)
            # new_total = check_total(new_lista, distance_matrix, cost_list)
            new_total = total_cost + rec_edge(random_indexes[first_ind], random_indexes[second_ind], lista, distance_matrix)

            if new_total < total_cost:
                new_lista = edge_swap_indices(lista, random_indexes[first_ind],random_indexes[second_ind])
                return new_lista, unvisited, new_total, False
    
    return lista, unvisited, total_cost, True



def find_best_edges(lista, total_cost, unvisited, distance_matrix, cost_list):

    # best = lista, unvisited, total_cost, True
    best = -1, -1 , total_cost, True

    n = len(lista)
    random_indexes = list(range(len(lista)))

    random.shuffle(random_indexes)

    
    for first_ind in range(len(lista)):
        for second_ind in range(first_ind+1, len(lista)):

            if abs(random_indexes[first_ind] - random_indexes[second_ind]) == 1:
                continue

            if random_indexes[first_ind] > random_indexes[second_ind]:
                _ = second_ind
                second_ind = first_ind
                first_ind = _
            
            # new_lista = deepcopy(lista)
            # new_total = check_total(new_lista, distance_matrix, cost_list)
            new_total = total_cost + rec_edge(random_indexes[first_ind], random_indexes[second_ind], lista, distance_matrix)

            if new_total < best[2]:
                best = random_indexes[first_ind],random_indexes[second_ind], new_total, False

    if not best[-1]:
        lista = edge_swap_indices(lista, best[0], best[1])

    return lista, unvisited, best[2], best[3]