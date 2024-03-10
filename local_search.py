import random
from utils import create_cur_tour_from_list, create_dist_matrix_and_cost, check_total, find_best, find_best_edges, plotMap, get_data, inter_swap_indexes, intra_swap_nodes

import numpy as np
import time
from tqdm import tqdm
import matplotlib.pyplot as plt

def repeat_ils(method, random_fill, ls, data, indices):
    start_time = time.time()
    total_cost = []
    best_cost = np.inf
    best_sol = -1
    iterations = []
    for i in tqdm(indices):
        random.seed(i)
        cost, sol, iteration = method(data, random_fill, ls)
        total_cost.append(cost)
        iterations.append(iteration)
        if cost<best_cost:
            best_cost = cost
            best_sol = sol
    end_time = time.time()
    execution_time = end_time - start_time
    execution_time = execution_time/len(indices)
    return total_cost, iterations, best_sol, execution_time

def destroy_reparir_candidate_ils(data, random_reconstruct, ls):
    best_cost = np.inf
    best_lista = []    
    msls_time = 45. #60.*15 # 15minutes
    start_time = time.time()
    totals = []
    n = len(data)
    distance_matrix, cost_list = create_dist_matrix_and_cost(data)
    n_candidates = round(n/2)+1
    candidates = np.zeros((n, n_candidates), dtype=int)
    for i in range(n):
        cand = np.argsort(distance_matrix[i] + cost_list[i])[:n_candidates+1]
        ind = np.where(cand == i)
        cand =np.delete(cand, ind)
        candidates[i] = cand[:n_candidates]

    nodes = list(range(0, n))
    random.shuffle(nodes)
    n_lista = round(float(n/2))
    lista = nodes[:n_lista]
    unvisited = nodes[n_lista:]
    total_cost = check_total(lista, distance_matrix, cost_list)

    repetitions = 2000
    ls_iterations = 0
    for n_perturbation in tqdm(range(repetitions)):
        if ls:
            for i in range(500):
                assert total_cost>0, f"Outside function{total_cost, i}"
                lista, unvisited, total_cost, terminate = find_best(lista, total_cost, unvisited, distance_matrix, exchange = "intra", cost_list=cost_list, candidates = candidates)

                lista, unvisited, total_cost, terminate = find_best_edges(lista, total_cost, unvisited, distance_matrix,cost_list=cost_list, candidates = candidates)

                if terminate:
                    break
            ls_iterations +=i
            if total_cost < best_cost:
                best_cost = total_cost
                best_lista = lista

            # print(time.time()- start_time, msls_time, total_cost)
            totals.append(total_cost)
        if time.time()-start_time > msls_time:
            # plt.plot(totals)
            # plt.plot()
            # print(best_cost)
            print("AAAAAAAAAAa", best_cost)
            if best_cost == np.inf:
                return total_cost, lista, ls_iterations/repetitions
            return best_cost, best_lista, ls_iterations/repetitions
        
        lista, unvisited, total_cost = distroy_and_repair(lista, unvisited, total_cost, distance_matrix, cost_list, random_fill=random_reconstruct)
     
    # return best_cost, best_lista, ls_iterations/repetitions


def distroy_and_repair(lista, unvisited, total_cost, distance_matrix, cost_list, random_fill, fraction = 0.3):
    n = len(lista)
    # print(total_cost, end = "   ")
    # random.seed(0)
    #30% of the list in one part is removed and replaced with random
    index_1 = random.randint(0,n)
    diff = round(fraction*n)
    index_2 = (index_1+diff)%n
    if index_1 > index_2:
        lista = lista[index_1:] + lista[:index_2] + lista[index_2:index_1]
    else:
        lista = lista[index_1:] + lista[:index_1]
    assert len(lista) == len(np.unique(lista)), f"JUST REORDERING {lista}, {len(lista), len(np.unique(lista))}"
    unvisited += lista[:diff]

    #random filling
    if random_fill:

        replacement = np.random.choice(unvisited, diff, replace = False)

        lista[:diff] = replacement
        for i in replacement:
            unvisited.remove(i)

        total_cost = check_total(lista, distance_matrix, cost_list)

        assert len(lista) == len(np.unique(lista)), f"AFTER REPLACEMENT {lista}, {len(lista), len(np.unique(lista))}\n{replacement}, "
    
    #use greedy cycle for this
    else:
        #plotMap(data, create_cur_tour_from_list(lista, distance_matrix, cost_list))

        lista = lista[diff:]
        #plotMap(data, create_cur_tour_from_list(lista, distance_matrix, cost_list))
        # print(total_cost, end = "  ")
        lista, unvisited = greedy_cycle(lista, unvisited, distance_matrix, cost_list)
        total_cost = check_total(lista, distance_matrix, cost_list)
        # print(total_cost)

    return lista, unvisited, total_cost

def greedy_cycle(lista, unvisited, distance_matrix, cost_list):
    # total_diff = 0
    first_length = len(lista)
    final_lenght = round(len(cost_list)/2)
    #plotMap(data, create_cur_tour_from_list(lista, distance_matrix, cost_list))
    for current_lenght in range(first_length, final_lenght):
        results = np.full([current_lenght, len(unvisited)],np.inf)
        for visited_index in range(current_lenght):
            for unvisited_index in range(len(unvisited)):
            
                cost_diff = \
                    -distance_matrix[lista[visited_index]][lista[(visited_index+1)%current_lenght]] \
                    + distance_matrix[lista[visited_index]][unvisited[unvisited_index]] \
                    + distance_matrix[unvisited[unvisited_index]][lista[(visited_index+1)%current_lenght]] \
                    + cost_list[unvisited[unvisited_index]]
                results[visited_index, unvisited_index] = cost_diff
        visited_index, unvisited_index = np.unravel_index(np.argmin(results), shape= [current_lenght, len(unvisited)])
        # total_diff += results[visited_index, unvisited_index]
        lista = lista[:visited_index+1] + [unvisited[unvisited_index]] + lista[visited_index+1:]
        unvisited.pop(unvisited_index)
    #plotMap(data, create_cur_tour_from_list(lista, distance_matrix, cost_list))
    
    return lista, unvisited#, total_diff

if "__main__" == __name__:
    data = get_data('TSPD.csv')[::]
    for i in range(len(data)):
        data[i][-1] = 0
    dist, cost = create_dist_matrix_and_cost(data)   
    total_cost, lista, iterations = destroy_reparir_candidate_ils(data, random_reconstruct = False, ls = False )

    plotMap(data, edges=create_cur_tour_from_list(lista, dist, cost))
    print(total_cost, lista, iterations)
    