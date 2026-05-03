import pandas as pd
import math
import argparse


def get_quartiles(data):
    data = [x for x in data if not math.isnan(x)]
    data = sorted(data)
    n = len(data)
    p = 0.5
    q2_i = (n-1)*p
    q2_idx = int(q2_i)
    fraction = q2_i - q2_idx
    if fraction != 0.0:
        q2 = data[q2_idx]+(data[q2_idx+1] - data[q2_idx])*fraction
    else:
        q2 = data[q2_idx]

    p = 0.25
    q1_i = (n-1)*p
    q1_idx = int(q1_i)
    fraction = q1_i - q1_idx
    if fraction != 0.0:
        q1 = data[q1_idx]+(data[q1_idx+1] - data[q1_idx])*fraction
    else:
        q1 = data[q1_idx]

    p = 0.75
    q3_i = (n-1)*p
    q3_idx = int(q3_i)
    fraction = q3_i - q3_idx
    if fraction != 0.0:
        q3 = data[q3_idx]+(data[q3_idx+1] - data[q3_idx])*fraction
    else:
        q3 = data[q3_idx]

    return q1, q2, q3


def describe(data):
    columns = []
    all_cloumns = data.columns
    for col in all_cloumns:
        if type(data[col][0]).__name__ != "str":
            columns.append(col)

    n_data = data[columns]
    descripe = pd.DataFrame(index=["count", "mean", "std", "min", "25%", "50%", "75%", "max", "nan"], columns=columns)
    for col in columns:
        count = sum([1 for i in range(n_data[col].size) if not math.isnan(n_data[col][i])])
        nan = sum([1 for x in n_data[col] if math.isnan(x)])
        if count > 0:
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
            descripe[col] = [count, mean, std, min, q1, q2, q3, max, nan]
    return descripe


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="describe data")
    parser.add_argument("file", help="CSV file to describe")
    args = parser.parse_args()

    file_name = args.file
    try:
        data = pd.read_csv(file_name)
    except Exception as e:
        print(f"Error reading file: {e}")
        exit(1)

    descripe = describe(data)
    print(descripe)
