import random
from utils import create_cur_tour_from_list, create_dist_matrix_and_cost, check_total, find_best, find_best_edges, plotMap, get_data
import numpy as np
import time
from tqdm import tqdm
random.seed(0)
def repeat_local_candidate(method, data, indices):
    start_time = time.time()
    total_cost = []
    best_cost = np.inf
    best_sol = -1
    best_ind = -1
    for i in tqdm(indices):
        cost, sol = method(data)
        total_cost.append(cost)
        if cost<best_cost:
            best_cost = cost
            best_sol = sol
            best_ind = i
    end_time = time.time()

    execution_time = end_time - start_time
    return total_cost, best_sol, best_ind, execution_time

def local_search_candidate(data):
    n = len(data)
    distance_matrix, cost_list = create_dist_matrix_and_cost(data)
    n_candidates = round(n/2)+1
    candidates = np.zeros((n, n_candidates), dtype=int)
    for i in range(n):
        cand = np.argsort(distance_matrix[i] + cost_list[i])[:n_candidates+1]
        ind = np.where(cand == i)
        cand =np.delete(cand, ind)
        candidates[i] = cand[:n_candidates]


    lista = list(range(0, len(data)))
    random.shuffle(lista)
    lista = lista[:round(float(len(lista))/2)]
    total_cost = check_total(lista, distance_matrix, cost_list)


    unvisited = [x for x in range(len(data))]
    
    for i in lista:
        unvisited.remove(i)

    for i in range(500):
        assert total_cost>0, f"Outside function{total_cost, i}"
        lista, unvisited, total_cost, terminate = find_best(lista, total_cost, unvisited, distance_matrix, exchange = "intra", cost_list=cost_list, candidates = candidates)

        lista, unvisited, total_cost, terminate = find_best_edges(lista, total_cost, unvisited, distance_matrix,cost_list=cost_list, candidates = candidates)

        if terminate:
            break
    return total_cost, lista



if "__main__" == __name__:
    data = get_data('TSPD.csv')
    # x = local_search_candidate(data)
    lista = [136, 73, 185, 132, 52, 8, 63, 82, 138, 21, 192, 196, 117, 71, 107, 12, 119, 59, 147, 159, 64, 129, 89, 58, 72, 85, 114, 150, 44, 162, 158, 67, 156, 91, 70, 51, 174, 161, 140, 188, 148, 141, 130, 13, 142, 53, 69, 113, 115, 40, 16, 18, 19, 190, 198, 135, 95, 172, 163, 182, 169, 66, 128, 5, 34, 179, 122, 143, 127, 24, 121, 31, 101, 38, 103, 131, 50, 152, 94, 112, 43, 116, 99, 0, 57, 137, 165, 37, 123, 134, 36, 25, 154, 88, 55, 153, 80, 157, 145, 79]
    dist, cost = create_dist_matrix_and_cost(data)
    plotMap(data, edges=create_cur_tour_from_list(lista, dist, cost))