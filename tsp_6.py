import csv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import csv
import random
from utils import *
from local_search import destroy_reparir_candidate_ils, repeat_ils
random.seed(0)
data = get_data('TSPD.csv')
random_list = [i for i in range(len(data))]
repetition = 20

print(len(data))
time = -1
result = dict()
for random_fill in [False]:
    for local_search in [True, False]:
        if random_fill and local_search is False:
           continue
        print(random_fill, local_search)
        costs, iterations, best_sol, time =  repeat_ils(destroy_reparir_candidate_ils, random_fill, local_search, data, list(range(repetition)))
        print(best_sol)
        title = f"random:{random_fill}_ls:{local_search}"
        best = f"{[int(x) for x in best_sol]}"
        result[title] = dict()
        result[title]["min"] = min(costs)
        result[title]["max"] = max(costs)
        result[title]["mean"] = np.mean(costs)
        result[title]["std"] = np.std(costs)
        result[title]['time'] = time
        result[title]['iterations'] = np.mean(iterations)
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
import tabulate
print(tabulate.tabulate(df_combined))