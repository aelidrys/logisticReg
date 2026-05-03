import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import argparse

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sys

def find_homogeneous_course(filename):
    try:
        df = pd.read_csv(filename)
        
        # Plotting
        plt.figure(figsize=(10, 6))
        
        sns.histplot(data=df, x="Care of Magical Creatures", hue='Hogwarts House', element="poly")
        
        plt.title(f"Score Distribution: Care of Magical Creatures")
        plt.xlabel("Score")
        plt.ylabel("Frequency")
        plt.show()

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    parcer = argparse.ArgumentParser(description="Histogram expect dataset in a .csv file")
    parcer.add_argument("file",  help="CSV file to describe")
    args = parcer.parse_args()
    file_name = args.file
    try:
        data = pd.read_csv(file_name)
    except Exception as e:
        print(f"Error reading file: {e}")
        exit(1)
    find_homogeneous_course(file_name)
