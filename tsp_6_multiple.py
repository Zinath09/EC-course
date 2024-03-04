import csv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import csv
import random
from utils import *
from local_search import multiple_local_search_candidate, repeat_local_candidate

random.seed(0)
data = get_data('TSPD.csv')
random_list = [i for i in range(len(data))]


print(len(data))

result = dict()
for start_solution in ['random']:
    for alg_type in ['steepest']:
        for neighbors in ['edges']:
            title = f"{'_'.join([start_solution, alg_type, neighbors])}"
            print("*"*10, title)
            costs, best_sol ,best_ind_random_random, time =  repeat_local_candidate(multiple_local_search_candidate,data, list(range(20)))
            best = f"{[int(x) for x in best_sol]}"
            result[title] = dict()
            result[title]["min"] = min(costs)
            result[title]["max"] = max(costs)
            result[title]["mean"] = np.mean(costs)
            result[title]["std"] = np.std(costs)
            result[title]['time'] = time
            result[title]["best"] = best

dfs = []

for key, value in result.items():
    df = pd.DataFrame(value, index = [key])
    df['main_key'] = key  # 
    dfs.append(df)

df_combined = pd.concat(dfs, ignore_index=True)

df_combined.set_index('main_key', inplace=True)
df_combined.to_csv('results.csv')
print("UPDATED")
