import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import argparse

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sys

def find_homogeneous_course(df, column):
    try:
        
        # Plotting
        plt.figure(figsize=(14, 8))
        
        sns.histplot(data=df, x=column, hue='Hogwarts House', element="poly")
        
        plt.title(f"Score Distribution: {column}")
        plt.xlabel("Score")
        plt.ylabel("Frequency")
        plt.show()

    except Exception as e:
        print(f"Error: {e}.")
        exit(1)

if __name__ == "__main__":
    parcer = argparse.ArgumentParser(description="Histogram expect dataset in a .csv file")
    parcer.add_argument("file",  help="CSV file to describe")
    parcer.add_argument("--column",  help="Column that will be used in histogram", required=False)
    args = parcer.parse_args()
    file_name = args.file
    column = args.column
 
    try:
        data = pd.read_csv(file_name)
    except Exception as e:
        print(f"Error reading file: {e}.")
        exit(1)

    columns = data.select_dtypes(include="float").columns
    if column is None:
        column = "Care of Magical Creatures"
    if column not in columns:
        print(f"Error: column '{column}' not a Hogwarts course.")
        exit(1)
    find_homogeneous_course(data, column)
