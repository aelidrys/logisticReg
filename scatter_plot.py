import pandas as pd
import  numpy as np
import matplotlib.pyplot as plt
import argparse


HOUSE_COLORS = {
    "Gryffindor": "#ae0001",
    "Hufflepuff": "#ecb939",
    "Ravenclaw": "#222f5b",
    "Slytherin": "#2a623d",
}

def load_data(filename):
    try:
        data = pd.read_csv(filename)
        return data
    except Exception as e:
        print(f"Error reading file: {e}")
        exit(1)

# Find the correlation between two features
def find_most_similar_pair(data):
    courses = data.select_dtypes(include=[np.number]).columns
    best = (None, None, 0.0)
    for i in range(len(courses)):
        for j in range(i + 1, len(courses)):
            corr = data[courses[i]].corr(data[courses[j]])
            if abs(corr) > abs(best[2]):
                best = (courses[i], courses[j], corr)
    return best


def create_scatter_plot(data):
    most_similar_pair = find_most_similar_pair(data)
    x_col = most_similar_pair[0]
    y_col = most_similar_pair[1]


    plt.figure(figsize=(15, 10))
    for house, color in HOUSE_COLORS.items():
        subset = data[data["Hogwarts House"] == house]
        plt.scatter(subset[x_col], subset[y_col], c=color, alpha=0.6, s=20, label=house)

    plt.title(f"Scatter Plot: {x_col} vs {y_col}")
    plt.xlabel(x_col, fontsize=15)
    plt.ylabel(y_col, fontsize=15)
    # increase font size
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a scatter plot from CSV data")
    parser.add_argument("filename", help="Path to the CSV file")

    args = parser.parse_args()
    file_name = args.filename

    data = load_data(file_name)
    create_scatter_plot(data)
