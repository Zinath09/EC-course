import csv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import csv
import random
from utils import *
from local_search import multiple_start_candidate_ils, perturbation_candidate_ils, repeat_ils
random.seed(0)
data = get_data('TSPD.csv')
random_list = [i for i in range(len(data))]
repetition = 2

print(len(data))
time = -1
result = dict()
for title in ["multiple_start"]:
    if title == "multiple_start":
        method = multiple_start_candidate_ils
    elif title == "perturbation":
        method = perturbation_candidate_ils
    print("*"*10, title)
    costs, iterations, best_sol, time =  repeat_ils(method, data, list(range(repetition)))
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
df_combined.to_csv('results_perm.csv')
print("UPDATED")
