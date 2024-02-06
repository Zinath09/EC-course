from CycleAlgorithm import CycleAlgorithm
import random
from utils import check_total, find_best, find_best_edges, find_first_better, find_first_better_edges, plotMap, get_data, return_statistic, present_statistic
import numpy as np
from algorithms import greedy_cycle
import time

def repeat_local(method, indices, data, start_solution, alg_type, exchange):
    start_time = time.time()
    total_cost = []
    best_cost = np.inf
    best_sol = -1
    best_ind = -1
    for i in indices:
        cost, sol = method(data, start_solution, alg_type, exchange)
        total_cost.append(cost)
        if cost<best_cost:
            best_cost = cost
            best_sol = sol
            best_ind = i
    end_time = time.time()

    execution_time = end_time - start_time
    return total_cost, best_sol, best_ind, execution_time


def local_search(data, start_solution, alg_type, exchange):

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
        total_cost, edges = greedy_cycle(distance_matrix, cost_list, n, round(n/2), 0)
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
        
        neighbors = ["nodes","edges"][random.randint(0,1)]

        if neighbors == "nodes":
            if alg_type == 'greedy':
                lista, unvisited, total_cost, terminate = find_first_better(lista, total_cost, unvisited, distance_matrix, exchange = exchange)
            elif alg_type == "steepest":
                lista, unvisited, total_cost, terminate = find_best(lista, total_cost, unvisited, distance_matrix, exchange = exchange)

        if neighbors == 'edges':
            if alg_type == 'greedy':
                lista, unvisited, total_cost, terminate = find_first_better_edges(lista, total_cost, unvisited, distance_matrix)
            elif alg_type == "steepest":
                lista, unvisited, total_cost, terminate = find_best_edges(lista, total_cost, unvisited, distance_matrix)
        if terminate:
            print("TERMINATED", i)
            break
        # print(total_cost, i)
    edges = alg.create_cur_tour_from_list(lista)
    # print(cost_list)
    # plotMap(data, edges)
    return total_cost, lista
