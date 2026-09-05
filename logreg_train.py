import numpy as np
import pandas as pd
import argparse
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

def load_data(filename):
    try:
        data = pd.read_csv(filename)
        data = data.drop(columns=["Index", "First Name", "Last Name"])
        return data
    except Exception as e:
        print(f"Error while loading data: {e}")
        exit(1)

def preprocess_data(data):
    columns = data.select_dtypes(include=[np.number]).columns.tolist()
    for col in columns:
        missing_values = data[col].isna().sum()
        if missing_values > 0:
            data[col] = data[col].fillna(data[col].mean())
    data = data.drop(columns=["Defense Against the Dark Arts"])
    data = pd.get_dummies(data, columns=["Best Hand"], drop_first=True)
    data["Birthday"] = pd.to_datetime(data["Birthday"], errors='coerce')
    data["Birthday"] = data["Birthday"].dt.strftime('%Y%m%d')
    label_encoder = LabelEncoder()
    data["Hogwarts House"] = label_encoder.fit_transform(data["Hogwarts House"])
    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a logistic regression model on CSV data")
    parser.add_argument("filename", help="Path to the CSV file")

    args = parser.parse_args()
    file_name = args.filename

    # load data
    data = load_data(file_name)

    # preprocess data
    data = preprocess_data(data)

    # split features and target variable
    X = data.drop(columns=["Hogwarts House"])
    y = data["Hogwarts House"]

    # split into training and testing sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # train logistic regression model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # make predictions
    y_pred = model.predict(X_val)
    # evaluate model
    print("Classification Report:")
    print(classification_report(y_val, y_pred))
    
    print("Confusion Matrix:")
    print(confusion_matrix(y_val, y_pred))