
import warnings
warnings.filterwarnings("ignore")
import argparse
import pandas as pd
import numpy as np
import pickle as pkl
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from logreg_train import load_data, preprocess_data, OneVsAllClassifier, log_reg



# Load the trained model
def load_params(model_path):
    try:
        with open(model_path, 'rb') as f:
            model_data = pkl.load(f)
            return model_data
    except Exception as e:
        print(f"Error while loading model: {e}")
        exit(1)


def load_weights(filename):
    try:
        weights_df = pd.read_csv(filename)
        return weights_df
    except Exception as e:
        print(f"Error while loading weights: {e}")
        exit(1)


def predict(weights_df, X, proba=False):
    y_pred = []
    for house in weights_df.columns:
        weights = weights_df[house].values.astype(float).reshape(-1, 1)
        y_house = X @ weights
        y_pred.append(y_house)
    y_pred_proba = np.hstack(y_pred)
    if proba:
        return y_pred_proba
    y_pred = np.argmax(y_pred_proba, axis=1).reshape(-1, 1)
    return y_pred


def test_with_validation(weights_df, params_dict):
    # Load and preprocess the data
    data = load_data('datasets/dataset_train.csv')
    data, label_encoder = preprocess_data(data)
    X = data.drop(columns=["Hogwarts House"], errors='ignore')
    y = data["Hogwarts House"].values
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=70, stratify=y)

    # scale features with standardization
    scaler = params_dict['scaler']
    X_val = scaler.transform(X_val)

    # Make predictions
    y_pred = predict(weights_df, X_val)
    y_pred_proba = predict(weights_df, X_val, proba=True)

    # evaluate model
    accuracy = accuracy_score(y_val, y_pred)
    print(f"Validation Accuracy: {accuracy:.4f}")

    y_val_onehot = OneHotEncoder(sparse_output=False).fit_transform(y_val.reshape(-1, 1))
    auc = roc_auc_score(y_val_onehot, y_pred_proba, multi_class='ovr')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict Hogwarts House using a trained logistic regression model")
    parser.add_argument("data_file", help="Path to the CSV file containing data for prediction")
    parser.add_argument("weights_path", help="Path to the CSV file containing data for prediction")
    args = parser.parse_args()

    # Load the trained model weights and params
    weights_df = load_weights(args.weights_path)
    params_path = 'logreg_params.pkl'
    params_dict = load_params(params_path)

    # Load and preprocess the data
    data = load_data(args.data_file)
    indexes = data['Index'].values
    means = params_dict['means']
    data, _ = preprocess_data(data, test=True, means=means)
    X_test = data.drop(columns=["Hogwarts House"], errors='ignore')

    # scale features with standardization
    scaler = params_dict['scaler']
    X_test = scaler.transform(X_test)

    # Make predictions
    y_pred = predict(weights_df, X_test)
    label_encoder = params_dict['label_encoder']
    hogwarts_house = label_encoder.inverse_transform(y_pred.flatten())

    result_df = pd.DataFrame({'Index': indexes, 'Hogwarts House': hogwarts_house})
    result_df.to_csv('houses.csv', index=False)


    