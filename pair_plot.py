import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import argparse
import warnings
warnings.filterwarnings("ignore")


def load_data(filename):
    try:
        data = pd.read_csv(filename)
        return data
    except Exception as e:
        print(f"Error while loading data: {e}")
        exit(1)


# Encoding categorical variables
def encode_categorical_variables(data, test=False):

    data["Birthday"] = pd.to_datetime(data["Birthday"], errors='coerce')
    data['Birth_year'] = data['Birthday'].dt.year
    columns_to_drop = ["Index", "First Name", "Last Name", "Defense Against the Dark Arts", "Birthday"]
    data = data.drop(columns=columns_to_drop, errors='ignore')
    data = pd.get_dummies(data, columns=["Best Hand"], drop_first=True)
    if test:
        data = data.drop(columns=["Hogwarts House"], errors='ignore')
    else:
        label_encoder = LabelEncoder()
        data["Hogwarts House"] = label_encoder.fit_transform(data["Hogwarts House"])
    return data


# display correlation heatmap
def show_correlation_heatmap(data):

    corr_matrix = data.corr()
    plt.figure(figsize=(16,10))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.subplots_adjust(left=0.21, right=0.96, top=0.96, bottom=0.23)
    plt.title("Correlation Heatmap")
    plt.show()


# display pair plot graph
def show_pair_plot(data):
    
    # Create pair plot
    print(f"Creating pair plot with features:\n{data.columns.tolist()}")
    sns.pairplot(data, hue="Hogwarts House")
    plt.subplots_adjust(left=0.05, right=0.96, top=0.96, bottom=0.05)
    plt.suptitle("Pair Plot of Features Colored by Hogwarts House classes", color="blue", fontsize=16)
    plt.show()



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a pair plot from CSV data")
    parser.add_argument("filename", help="Path to the CSV file")

    args = parser.parse_args()
    file_name = args.filename

    # load data
    data = load_data(file_name)

    # encode categorical columns
    data = encode_categorical_variables(data)

    # show Pair Plot
    show_pair_plot(data)


