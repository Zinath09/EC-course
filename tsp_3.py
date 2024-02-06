import csv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import csv
import random
from utils import *
from local_search import local_search, repeat_local
# from algorithms import greedy_cycle
from CycleAlgorithm import CycleAlgorithm
import json
random.seed(0)

start_solution = "random"
start_solution = "best"
alg_type = "greedy"
# alg_type = "steepest"

exchange = "intra" #poza
# exchange = "inter" #wewn

data = get_data('TSPD.csv')
result = dict()
for start_solution in ['random', 'best']:
    for alg_type in ['greedy', 'steepest']:
        for exchange in ['intra', 'inter']:
            random_list = [0,1]#random.randint(0,len(data)-1) for i in range(100)]

            costs, best_sol ,best_ind_random_random, time =  repeat_local(local_search, random_list,data, start_solution, alg_type, exchange)
            best = f"{[int(x) for x in best_sol]}"
            print(best)
            type = f"{'_'.join([start_solution, alg_type, exchange])}"
            result[type] = dict()
            result[type]["min"] = min(costs)
            result[type]["max"] = max(costs)
            result[type]["mean"] = np.mean(costs)
            result[type]["std"] = np.std(costs)
            result[type]['time'] = time
            result[type]["best"] = best


dfs = []

# Iterate through each dictionary in the list
for key, value in result.items():
    df = pd.DataFrame(value, index = [key])
    df['main_key'] = key  # Add a column for the main key
    dfs.append(df)

# Concatenate the individual DataFrames into one DataFrame
df_combined = pd.concat(dfs, ignore_index=True)

# Set the main key as the index (optional)
df_combined.set_index('main_key', inplace=True)
df_combined.to_csv('results.csv')
# for i in range(len(data)):
#     data[i][2] = 0
# data = np.array([(0, 0, 0), (5, 3,0 ), (1, 4,0), (3, 1,0), (7, 3,0), (2,5,0), (4,4, 0)]) * 
