import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math


def get_quartiles(data):
    data = sorted(data)
    data = [x for x in data if not math.isnan(x)]
    n = len(data)
    q2_idx = int(n/2)
    if n % 2 == 0:
        lower_half = data[:q2_idx]
        q2 = (data[q2_idx]+data[q2_idx-1])/2
    else:
        lower_half = data[:q2_idx-1]
        q2 = data[q2_idx]

    q1_idx = int(len(lower_half)/2)
    if len(lower_half) % 2 == 0:
        q1 = (lower_half[q1_idx]+lower_half[q1_idx-1])/2
    else:
        q1 = lower_half[q1_idx]

    upper_half = data[q2_idx:]
    q3_idx = int(len(upper_half)/2)
    if len(upper_half) % 2 == 0:
        q3 = (upper_half[q3_idx]+upper_half[q3_idx-1])/2
    else:
        q3 = upper_half[q3_idx]

    return q1, q2, q3


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
descripe = pd.DataFrame(index=["count", "mean", "std", "min", "25%", "50%", "75%", "max"], columns=columns)
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
    q1, q2, q3 = get_quartiles(n_data[col])
    descripe[col] = [count, mean, std, min, q1, q2, q3, max]

print(descripe)