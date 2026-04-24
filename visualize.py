import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

data = pd.read_csv("./data.csv")
print(data.describe())

print("\n--------------------------------------------\n")
print("\n--------------------------------------------\n")

columns = []
all_cloumns = data.columns
for col in all_cloumns:
    if type(data[col][0]).__name__ != "str":
        columns.append(col)

n_data = data[columns]
descripe = pd.DataFrame(index=["count", "mean", "std", "min", "max"], columns=columns)
for col in columns:
    count = sum([1 for i in range(n_data[col].size) if not math.isnan(n_data[col][i])])
    mean = sum([n_data[col][i] for i in range(n_data[col].size) if not math.isnan(n_data[col][i])]) / count
    min = math.inf 
    max = -math.inf
    std = 0
    for i in range(n_data[col].size):
        if not math.isnan(n_data[col][i]):
            if n_data[col][i] < min:
                min = n_data[col][i]
            if n_data[col][i] > max:
                max = n_data[col][i]
            std += (n_data[col][i] - mean) ** 2
    std = math.sqrt(std / (count-1))
    descripe[col] = [count, mean, std, min, max]

print(descripe)