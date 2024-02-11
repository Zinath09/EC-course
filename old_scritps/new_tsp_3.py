import csv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import csv
import random
from utils import *
from algorithms import *
from Algorithm import Algorithm
from GreedyAlgorithm import GreedyAlgorithm


start_solution = "random"
# start_solution = "best"
alg_type = "greedy"
alg_type = "steepest"
# neighbors = 'edges'
# neighbors = 'nodes' 


data = get_data('TSPD.csv')
data = np.array([(0, 0), (5, 3), (1, 4), (3, 1), (7, 3), (2,5), (4,4)]) * 100

# if alg_type == "greedy":
alg = GreedyAlgorithm(data)


alg.starting_solution(1)

# print(alg.cur_tour)

while round(len(data)/2)>len(alg.cur_tour):
    previous_index, new_node = alg.next_node()
    alg.add_to_tour(alg.endings[previous_index], new_node)
    plotMap(data, alg.cur_tour, cost = False)
alg.add_to_tour(new_node, alg.endings[(previous_index+1)%1])
print( alg.cur_tour)
plotMap(data, alg.cur_tour, cost = False)

