import matplotlib.pyplot as plt
import numpy as np
import random
from copy import deepcopy
import csv 

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

def get_min_value(array):
    return np.min(array)

def get_min_index(array):
    min_index = np.argmin(array)
    min  = array[min_index]
    return min, min_index

def Euclidian_distance(coor_1, coor_2):
    x1, y1 = coor_1
    x2, y2 = coor_2
    return int(((x2 -x1)**2 + (y2-y1)**2 )**(1/2))

def get_dist_matrix_and_cost(data, cost = True):
    NR_NODES = len(data)
    distance_matrix = np.zeros((NR_NODES,NR_NODES))
    cost_list = np.zeros(NR_NODES)
    for i in range(NR_NODES):
        for j in range(i,NR_NODES):
            dist = Euclidian_distance(data[i][:2],data[j][:2])
            if cost:
                cost_list[i] = data[i][2]
            distance_matrix[i][j] = dist
            distance_matrix[j][i] = dist
            distance_matrix[i][i] = np.inf
        
    return distance_matrix, cost_list


def add_to_cycle(edge, new_n, dist_m,edges, cost_list):
    #
    start_n = edge[0]
    end_n = edge[1]
    # cost(start-end) + cost(start, new) + cost(end,new) + cost(new)

    cost_diff = -dist_m[start_n][ end_n] + dist_m[start_n][ new_n] + dist_m[new_n][end_n] + cost_list[new_n]
    edges.remove(edge)
    edges.append([start_n, new_n])
    edges.append([new_n, end_n])
    return edges, cost_diff

def cound_cost_diff_cycle(edge, new_n, dist_m, cost_list):
    start_n = edge[0]
    end_n = edge[1]
    # cost(start-end) + cost(start, new) + cost(end,new) + cost(new)print(distance_matrix[:10][:10])
    cost_diff = -dist_m[start_n][ end_n] + dist_m[start_n][ new_n] + dist_m[new_n][end_n] + cost_list[new_n]

    return cost_diff
def new_get_min_index(array):
    return np.argmin(array)

def repeat(method, indices, distance_matrix, cost_list, NR_NODES, HALF_NODES):
    total_cost = []
    best_cost = np.inf
    best_sol = -1
    best_ind = -1
    for i in indices:
        cost, sol = method(distance_matrix, cost_list, NR_NODES, HALF_NODES, i)
        total_cost.append(cost)
        if cost<best_cost:
            best_cost = cost
            best_sol = sol
            best_ind = i
    return total_cost, best_sol, best_ind

def present_statistic(list):
    res = return_statistic(list)
    print("MIN: ",res[0])
    print("MAX: ",res[1])
    print("AVG: ",res[2])
    print("STD: ",res[3])

def return_statistic(list):
    return min(list), max(list), np.mean(list),  np.std(list)

def count_cost_diff_cycle(edge, new_n, dist_m, cost_list):
    start_n = edge[0]
    end_n = edge[1]
    # cost(start-end) + cost(start, new) + cost(end,new) + cost(new)print(distance_matrix[:10][:10])
    cost_diff = - dist_m[start_n][ end_n] + dist_m[start_n][ new_n] + dist_m[new_n][end_n] +cost_list[new_n]
    # assert cost_diff>0, f'{cost_diff, - dist_m[start_n][ end_n], dist_m[start_n][ new_n], dist_m[new_n][end_n]}' #
    return cost_diff

def create_regret_matrix(non_visited, cur_tour, dist_m, cost_list): #cur_tour = edges
    reg_matrix = np.zeros((len(dist_m),len(cur_tour)))
    for new_node in non_visited:
        for i,edge in enumerate(cur_tour):
            reg_matrix[new_node][i]=count_cost_diff_cycle(edge, new_node, dist_m, cost_list)

    return reg_matrix

def return_biggest_regret(matrix):
    min_values_for_rows = np.min(matrix, axis=1)
    # print("min_values_for_rows",min_values_for_rows)
    rescue_node = np.argmax(min_values_for_rows, axis=0) #najlepiej ratować 4 index
    # print("City with bigest regret",rescue_node)
    rescueing_node = np.argmin(matrix[rescue_node])# def return_max_from_min_rows_regret(matrix):
    # print("Rescueing edge index to modify: ",rescueing_node)
    return rescue_node, rescueing_node

def get_data(path):
    
    with open(path, newline='') as csvfile:
        data = list(csv.reader(csvfile, delimiter=';'))
        for item in range(len(data)):
            i = data[item]
            data[item] = [int(x) for x in i]
    return data


def inter_swap_nodes(node_1, node_2, lista):
    index_1 = lista.index(node_1)
    index_2 = lista.index(node_2)
    lista[index_1] = node_2
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

from recalculate import rec_inter_node, rec_edge, rec_intra_node
def find_first_better(lista, total_cost, unvisited, distance_matrix, exchange = "intra", cost_list = []):
    random.shuffle(unvisited)
    
    random_lista_indexes = list(range(len(lista)))
    random.shuffle(random_lista_indexes)
    random_lista = [lista[x] for x in random_lista_indexes] #po prostu shuffle lista

    # n = len(random_lista)
    if exchange == 'inter':
        for first_ind in range(len(lista)):
            for second_ind in range(first_ind+1, len(lista)):
                # new_lista = inter_swap_nodes(random_lista[first_ind], random_lista[second_ind], deepcopy(lista))
                # new_total = check_total(new_lista, distance_matrix, cost_list)
                new_total = total_cost + rec_inter_node(random_lista_indexes[first_ind], random_lista_indexes[second_ind], lista, distance_matrix)
                
                if new_total < total_cost:
                    new_lista = inter_swap_nodes(random_lista[first_ind], random_lista[second_ind], lista)

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


def find_best(lista, total_cost, unvisited, distance_matrix, exchange = 'intra', cost_list = [],alg = None):
    random_lista_indexes = list(range(len(lista)))
    
    random.shuffle(random_lista_indexes)
    random_lista = [lista[x] for x in random_lista_indexes] #po prostu shuffle lista
    best = -1, -1, total_cost, True

    if exchange == 'inter':
        for first_ind in range(len(lista)):
            for second_ind in range(first_ind+1, len(lista)):
                # new_lista = inter_swap_nodes(random_lista[first_ind], random_lista[second_ind], deepcopy(lista))
                # new_total = check_total(new_lista, distance_matrix, cost_list)
                new_total = total_cost + rec_inter_node(random_lista_indexes[first_ind], random_lista_indexes[second_ind], lista, distance_matrix)
                if new_total < best[2]:
                    print("delta",rec_inter_node(random_lista_indexes[first_ind], random_lista_indexes[second_ind], lista, distance_matrix) , total_cost, new_total, first_ind, second_ind)

                    best = random_lista_indexes[first_ind], random_lista_indexes[second_ind], new_total, False

        visited_node_index, unvisited_node, new_total, terminate = best
        if terminate:
            return lista, unvisited, total_cost, terminate
        else:
            new_lista = inter_swap_nodes(random_lista[first_ind], random_lista[second_ind], lista)
            print(len(np.unique(new_lista)))
            # plotMap(alg.data, alg.create_cur_tour_from_list(new_lista))
            return new_lista, unvisited, new_total, terminate

    elif exchange == 'intra':
        for unvisited_node in unvisited:
            for visited_node_index in random_lista_indexes:
                # new_lista, new_unvisited = intra_swap_nodes(visited_node_index, unvisited_node, deepcopy(lista), deepcopy(unvisited))
                # new_total = check_total(new_lista, distance_matrix, cost_list)
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