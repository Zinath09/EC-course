import random
from utils import create_cur_tour_from_list, create_dist_matrix_and_cost, check_total, find_best, find_best_edges, plotMap, get_data, edge_swap_indices, intra_swap_nodes

import numpy as np
import time
from tqdm import tqdm
import matplotlib.pyplot as plt

random.seed(0)
def repeat_ils(method, data, indices):
    start_time = time.time()
    total_cost = []
    best_cost = np.inf
    best_sol = -1
    iterations = []
    for i in tqdm(indices):
        cost, sol, iteration = method(data)
        total_cost.append(cost)
        iterations.append(iteration)
        if cost<best_cost:
            best_cost = cost
            best_sol = sol
    end_time = time.time()
    execution_time = end_time - start_time
    execution_time = execution_time/len(indices)
    return total_cost, iterations, best_sol, execution_time

def multiple_start_candidate_ils(data):

    #summary 
    best_cost = np.inf
    best_lista = []    
    totals = []

    #data
    n = len(data)
    distance_matrix, cost_list = create_dist_matrix_and_cost(data)

    #candidate matrix
    n_candidates = n-1#round(n/2)+1
    candidates = np.zeros((n, n_candidates), dtype=int)
    for i in range(n):
        cand = np.argsort(distance_matrix[i] + cost_list[i])#[:n_candidates+1]
        ind = np.where(cand == i)
        cand =np.delete(cand, ind)
        candidates[i] = cand#[:n_candidates]

    for n_iteration in tqdm(range(200)):
        #prepare random list of visited nodes and unvisited nodes
        lista = list(range(0, len(data)))
        random.shuffle(lista)
        lista = lista[:round(float(len(lista))/2)]
        total_cost = check_total(lista, distance_matrix, cost_list)
        unvisited = [x for x in range(len(data))]
        for i in lista:
            unvisited.remove(i)

        for i_ls in range(500):
            assert total_cost>0, f"Outside function{total_cost, i_ls}"
            lista, unvisited, total_cost, terminate = find_best(lista, total_cost, unvisited, distance_matrix, exchange = "intra", cost_list=cost_list, candidates = candidates)
            lista, unvisited, total_cost, terminate = find_best_edges(lista, total_cost, unvisited, distance_matrix,cost_list=cost_list, candidates = candidates)
            if terminate:
                break
        if total_cost < best_cost:
            best_cost = total_cost
            best_lista = lista
        # print(time.time()- start_time, msls_time, total_cost)
        totals.append(total_cost)
        # print(totals)
    # plt.plot(totals)
    # plt.show()
    return best_cost, best_lista, i_ls

def perturbation_candidate_ils(data):
    msls_time = 46. # 5minutes

    #summary 
    best_cost = np.inf
    best_lista = []    
    start_time = time.time()
    totals = []

    #data
    n = len(data)
    distance_matrix, cost_list = create_dist_matrix_and_cost(data)

    #candidate matrix
    n_candidates = n-1#round(n/2)+1
    candidates = np.zeros((n, n_candidates), dtype=int)
    for i in range(n):
        cand = np.argsort(distance_matrix[i] + cost_list[i])#[:n_candidates+1]
        ind = np.where(cand == i)
        cand =np.delete(cand, ind)
        candidates[i] = cand#[:n_candidates]

    #prepare random list of visited nodes and unvisited nodes
    lista = list(range(0, len(data)))
    random.shuffle(lista)
    lista = lista[:round(float(len(lista))/2)]
    total_cost = check_total(lista, distance_matrix, cost_list)
    unvisited = [x for x in range(len(data))]
    for i in lista:
        unvisited.remove(i)

    for n_iteration in tqdm(range(10000)):
        lista, unvisited = perturbate(lista, unvisited)
        total_cost = check_total(lista, distance_matrix, cost_list)

        for i_ls in range(500):
            assert total_cost>0, f"Outside function{total_cost, i_ls}"
            lista, unvisited, total_cost, terminate = find_best(lista, total_cost, unvisited, distance_matrix, exchange = "intra", cost_list=cost_list, candidates = candidates)
            lista, unvisited, total_cost, terminate = find_best_edges(lista, total_cost, unvisited, distance_matrix,cost_list=cost_list, candidates = candidates)
            if terminate:
                break
        if total_cost < best_cost:
            best_cost = total_cost
            best_lista = lista
        # print(time.time()- start_time, msls_time, total_cost)
        totals.append(total_cost)
        # print(totals)
        if time.time()-start_time > msls_time:
            # plt.plot(totals)
            # plt.plot()
            # print(best_cost)
            return best_cost, best_lista, i_ls

    return best_cost, best_lista,i_ls

def perturbate(lista, unvisited):
    n = len(lista)
    n_perturbation = round(0.15 * n)
    node_indices = np.random.choice(list(range(0,n)), n_perturbation)
    nodes_2_indices = np.random.randint(0,2*n, n_perturbation)
    
    for p in range(n_perturbation):
        is_visited = nodes_2_indices[p] < n
        if is_visited:
            #preparation for edge exchange
            first_ind = node_indices[p]
            second_ind = nodes_2_indices[p]
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
            if  diff == 1 or diff == n-1 or diff==0:
                continue
            lista = edge_swap_indices(lista,true_first_ind, true_second_ind)
        else:
            lista, unvisited = intra_swap_nodes(node_indices[p], unvisited[nodes_2_indices[p]-n], lista, unvisited)
    return lista, unvisited



if "__main__" == __name__:
    data = get_data('TSPD.csv')[::]
    dist, cost = create_dist_matrix_and_cost(data)   
    total_cost, lista, i_ls = multiple_start_candidate_ils(data)

    # total_cost, lista, i_ls =perturbation_candidate_ils(data)
    plotMap(data, edges=create_cur_tour_from_list(lista, dist, cost))
    print(total_cost, lista, i_ls)
    