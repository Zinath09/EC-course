import random
from utils import create_cur_tour_from_list, create_dist_matrix_and_cost, check_total, find_best, find_best_edges, plotMap, get_data
import numpy as np
import time
from tqdm import tqdm
import matplotlib.pyplot as plt

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


def iterated_local_search_candidate(data):
    best_cost = np.inf
    best_lista = []    
    msls_time = 60.*15 # 15minutes
    start_time = time.time()
    totals = []
    for n_perturbation in tqdm(range(2000)):
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
        else:
            lista, unvisited = perturbate(lista, unvisited)
            total_cost = check_total(lista, dist, cost)

    return best_cost, best_lista


def perturbate(lista,unvisited):#, , distance_matrix, cost_list    
    n = len(lista)
    unvisited = list(set(range(2*2)) - set(lista))

    #30% of the list in one part is removed and replaced with random
    random.seed()
    index_1 = random.randint(0,n)
    diff = round(0.3*n)
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
    replacement = np.random.choice(unvisited, diff, replace = False)
    # print("replacement:", lista[:diff])

    lista[:diff] = replacement
    # print([(i,x) for i, x in enumerate(lista) if lista.count(x) > 1])
    for i in replacement:
        unvisited.remove(i)
        
    assert len(lista) == len(np.unique(lista)), f"AFTER REPLACEMENT {lista}, {len(lista), len(np.unique(lista))}\n{replacement}, "
    return lista, unvisited

    
def multiple_local_search_candidate(data, rep = 200, start = []):
    best_cost = np.inf
    best_lista = []    
    for repetition in tqdm(range(rep)):
        n = len(data)
        distance_matrix, cost_list = create_dist_matrix_and_cost(data)
        n_candidates = round(n/2)+1
        candidates = np.zeros((n, n_candidates), dtype=int)
        for i in range(n):
            cand = np.argsort(distance_matrix[i] + cost_list[i])[:n_candidates+1]
            ind = np.where(cand == i)
            cand =np.delete(cand, ind)
            candidates[i] = cand[:n_candidates]

        if start!=[]:
            lista = start
        else:
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

        if total_cost < best_cost:
            best_cost = total_cost
            best_lista = lista
    return best_cost, best_lista



if "__main__" == __name__:
    data = get_data('TSPD.csv')
    # total_cost, lista = multiple_local_search_candidate(data,rep = 1)
    # print("total_cost", total_cost)
    dist, cost = create_dist_matrix_and_cost(data)   
    # unvisited = set(list(range(0,len(data)))) - set(lista)
    total_cost, lista =iterated_local_search_candidate(data)
    # for i in range(50):
    #     perturbated_lista, unvisited = perturbate(lista,list(unvisited))
    # # plotMap(data, edges=create_cur_tour_from_list(perturbated_lista, dist, cost))
    #     total_cost, lista = multiple_local_search_candidate(data, rep = 1, start=perturbated_lista)
    #     print("total_cost", total_cost)
    plotMap(data, edges=create_cur_tour_from_list(lista, dist, cost))

    # perturbated_lista, unvisited = perturbate(list(range(0,100)),list(range(100,200)))
    # print(len(np.unique(perturbated_lista)))
