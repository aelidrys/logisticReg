import pandas as pd
import matplotlib.pyplot as plt
import argparse


def load_data(filename):
    try:
        data = pd.read_csv(filename)
        return data
    except Exception as e:
        print(f"Error reading file: {e}")
        exit(1)


def create_scatter_plot(data):
    plt.figure(figsize=(10, 8))
    x_column = "Astronomy"
    y_column = "Defense Against the Dark Arts"
    plt.scatter(data[x_column], data[y_column], alpha=0.5)
    plt.title(f"Scatter Plot: {x_column} vs {y_column}")
    plt.xlabel(x_column)
    plt.ylabel(y_column)
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a scatter plot from CSV data")
    parser.add_argument("filename", help="Path to the CSV file")

    args = parser.parse_args()
    file_name = args.filename

    data = load_data(file_name)
    create_scatter_plot(data)
