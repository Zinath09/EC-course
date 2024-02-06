from CycleAlgorithm import CycleAlgorithm
import random
from utils import check_total, find_best, find_best_edges, find_first_better, find_first_better_edges, plotMap, get_data, return_statistic, present_statistic
import numpy as np
from algorithms import greedy_cycle
import time

def repeat_local(method, indices, data, start_solution, alg_type, neighbors):
    start_time = time.time()
    total_cost = []
    best_cost = np.inf
    best_sol = -1
    best_ind = -1
    for i in indices:
        cost, sol = method(data, start_solution, alg_type, neighbors, i)
        total_cost.append(cost)
        if cost<best_cost:
            best_cost = cost
            best_sol = sol
            best_ind = i
    end_time = time.time()

    execution_time = end_time - start_time
    return total_cost, best_sol, best_ind, execution_time


def local_search(data, start_solution, alg_type, neighbors, start_index = 0):

    n = len(data)
    alg = CycleAlgorithm(data)

    distance_matrix = alg.node_distances
    cost_list = alg.cost_list


    if start_solution == "random":
        lista = list(range(0, len(data)))
        random.shuffle(lista)
        lista = lista[:round(float(len(lista))/2)]
        alg.create_cur_tour_and_update(lista)
        total_cost = check_total(lista, distance_matrix)

    if start_solution =="best":
        total_cost, edges = greedy_cycle(distance_matrix, cost_list, n, round(n/2), start_index)
        lista = edges[0]
        list_1, list_2 = zip(*edges)
        while True:
            next = list_2[list_1.index(lista[-1])]
            if next == lista[0]:
                break
            lista.append(next)

    unvisited = [x for x in range(len(data))]
    for i in lista:
        unvisited.remove(i)
    for i in range(800):
        if alg_type == "steepest":

            lista, unvisited, total_cost, terminate = find_best(lista, total_cost, unvisited, distance_matrix, exchange = "inter")

            if neighbors == "nodes":
                lista, unvisited, total_cost, terminate = find_best(lista, total_cost, unvisited, distance_matrix, exchange = "intra")

            elif neighbors == 'edges':
                lista, unvisited, total_cost, terminate = find_best_edges(lista, total_cost, unvisited, distance_matrix)
                                    

        elif alg_type == "greedy":
            first_exchange = ["intra","inter"][random.randint(0,1)]
            if first_exchange == 'inter':
                    lista, unvisited, total_cost, terminate = find_first_better(lista, total_cost, unvisited, distance_matrix, exchange = "inter")

            if neighbors == "nodes":
                lista, unvisited, total_cost, terminate = find_first_better(lista, total_cost, unvisited, distance_matrix, exchange = "intra")

            elif neighbors == 'edges':
                lista, unvisited, total_cost, terminate = find_first_better_edges(lista, total_cost, unvisited, distance_matrix)

            if first_exchange != 'inter':
                    lista, unvisited, total_cost, terminate = find_first_better(lista, total_cost, unvisited, distance_matrix, exchange = "inter")


        if terminate:
            print(f"{'_'.join([start_solution, alg_type, neighbors])}",i)
            break
        # print(total_cost, i)
    edges = alg.create_cur_tour_from_list(lista)
    # print(cost_list)
    # plotMap(data, edges)
    return total_cost, lista




data = get_data('TSPD.csv')
start_solution = "random"
start_solution = "best"
alg_type = "greedy"
# alg_type = "steepest"

neighbors = "nodes" #poza
# neighbors = "edges" #wewn

local_search(data, start_solution, alg_type, neighbors)