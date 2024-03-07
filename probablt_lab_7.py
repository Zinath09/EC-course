import random
from utils import create_cur_tour_from_list, create_dist_matrix_and_cost, check_total, find_best, find_best_edges, plotMap, get_data, inter_swap_indexes, intra_swap_nodes

import numpy as np
import time
from tqdm import tqdm
import matplotlib.pyplot as plt

def destroy_reparir_candidate_ils(data, random_reconstruct = True):
    best_cost = np.inf
    best_lista = []    
    msls_time = 60.*15 # 15minutes
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

    repetitions = 20
    ls_iterations = 0
    for n_perturbation in tqdm(range(repetitions)):

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
            plt.plot(totals)
            plt.plot()
            print(best_cost)
            return best_cost, best_lista
        elif random_reconstruct:
            # plotMap(data, create_cur_tour_from_list(lista, distance_matrix, cost_list))
            print(total_cost, end = "  ")
            lista, unvisited = distroy_and_repair(lista, unvisited, random_fill=True)
            # plotMap(data, create_cur_tour_from_list(lista, distance_matrix, cost_list))
            total_cost = check_total(lista, distance_matrix, cost_list)
            print(total_cost)

    return best_cost, best_lista, ls_iterations/repetitions


def distroy_and_repair(lista, unvisited, random_fill = True, fraction = 0.3):#, , distance_matrix, cost_list    
    n = len(lista)
    #30% of the list in one part is removed and replaced with random
    random.seed()
    index_1 = random.randint(0,n)
    diff = round(fraction*n)
    index_2 = (index_1+diff)%n
    # print(index_1, index_2)
    if index_1 > index_2:
        lista = lista[index_1:] + lista[:index_2] + lista[index_2:index_1]
    else:
        lista = lista[index_1:] + lista[:index_1]
    assert len(lista) == len(np.unique(lista)), f"JUST REORDERING {lista}, {len(lista), len(np.unique(lista))}"
    # print("lista:", lista)
    # print("to change:", lista[:diff])
    unvisited += lista[:diff]

    if random_fill:
        replacement = np.random.choice(unvisited, diff, replace = False)
        # print("replacement:", lista[:diff])

        lista[:diff] = replacement
        # print([(i,x) for i, x in enumerate(lista) if lista.count(x) > 1])
        for i in replacement:
            unvisited.remove(i)
            
        assert len(lista) == len(np.unique(lista)), f"AFTER REPLACEMENT {lista}, {len(lista), len(np.unique(lista))}\n{replacement}, "
    else:
        pass
    return lista, unvisited

    


if "__main__" == __name__:
    data = get_data('TSPD.csv')[::]
    for i in range(len(data)):
        data[i][-1] = 0
    dist, cost = create_dist_matrix_and_cost(data)   
    total_cost, lista, iterations = destroy_reparir_candidate_ils(data, random_reconstruct = True )

    plotMap(data, edges=create_cur_tour_from_list(lista, dist, cost))
    print(total_cost, lista, iterations)
    