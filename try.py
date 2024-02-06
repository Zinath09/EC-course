import pandas as pd
from tabulate import tabulate
print(tabulate(pd.read_csv('results.csv')))