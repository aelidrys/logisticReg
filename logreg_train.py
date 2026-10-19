import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import argparse
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from describe import fn_describe
import pickle as pkl


def load_data(filename):
    try:
        data = pd.read_csv(filename)
        return data
    except Exception as e:
        print(f"Error while loading data: {e}")
        exit(1)


def preprocess_data(data, test=False, means=None):

    data["Birthday"] = pd.to_datetime(data["Birthday"], errors='coerce')
    columns_to_drop = ["Index", "First Name", "Last Name", "Defense Against the Dark Arts", 'Birthday', 'Best Hand', 'Arithmancy', 'Care of Magical Creatures']
    data = data.drop(columns=columns_to_drop, errors='ignore')

    columns = data.select_dtypes(include=[np.number]).columns.tolist()
    columns.remove('Hogwarts House') if 'Hogwarts House' in columns else None
    if means is None:
        describe_df = fn_describe(data)
        means = describe_df.loc["mean"]
    for col in columns:
        missing_values = data[col].isna().sum()
        if missing_values > 0:
            data[col] = data[col].fillna(means.loc[col])

    if test:
        data = data.drop(columns=["Hogwarts House"], errors='ignore')
        return data, None
    else:
        label_encoder = LabelEncoder().fit(data["Hogwarts House"])
        data["Hogwarts House"] = label_encoder.transform(data["Hogwarts House"]) # {'Gryffindor': 0, 'Hufflepuff': 1, 'Ravenclaw': 2, 'Slytherin': 3}

    return data, {'label_encoder': label_encoder, 'means': means}


class OneVsAllClassifier():
    def __init__(self, batch_size=128, lr=0.01, epochs=100, verbose=False):
        self.lr = lr
        self.epochs = epochs
        self.verbose = verbose
        self.batch_size = batch_size
        self.classifiers = {}

    def fit(self, X, y):
        classes = np.unique(y)
        clfs_wieghts = []
        for cl in classes:
            y_binary = (y == cl).astype(int)
            model = log_reg(n_features=X.shape[1], batch_size=self.batch_size, lr=self.lr, epochs=self.epochs, verbose=self.verbose)
            model.fit(X, y_binary)
            clfs_wieghts.append(model.wieghts.flatten())
            self.classifiers[cl] = model
        houses = ['Gryffindor', 'Hufflepuff', 'Ravenclaw', 'Slytherin']
        dict_ = {houses[i]: clfs_wieghts[i] for i in range(len(houses))}
        wieghts_df = pd.DataFrame(dict_)
        return wieghts_df


    def predict(self, X):
        y_probas = []
        for cl, model in self.classifiers.items():
            y_proba = model.predict_proba(X)
            y_probas.append(y_proba)

        y_proba_all = np.hstack(y_probas)
        y_pred = np.argmax(y_proba_all, axis=1)
        return y_pred.reshape(-1, 1)

    def predict_proba(self, X):
        y_probas = []
        for cl, model in self.classifiers.items():
            y_proba = model.predict_proba(X)
            y_probas.append(y_proba)

        y_proba_all = np.hstack(y_probas)
        y_proba = np.max(y_proba_all, axis=1)
        return y_proba.reshape(-1, 1)
        
    def predict_proba_all(self, X):
        y_probas = []
        for cl, model in self.classifiers.items():
            y_proba = model.predict_proba(X)
            y_probas.append(y_proba)

        y_proba_all = np.hstack(y_probas)
        return y_proba_all


class log_reg():

    def __init__(self, n_features, batch_size=128, lr=0.01, epochs=1000, verbose=False):
        self.lr = lr
        self.epochs = epochs
        self.verbose = verbose
        self.batch_size = batch_size
        limit = 1 / np.sqrt(n_features)
        self.wieghts = np.random.uniform(-limit, limit, (n_features, 1))

    def sigmiod(self, z):
        return 1 / (1 + np.exp(-z))

    def log_loss(self, t, y):
        return - np.mean(t * np.log(y) + (1 - t) * np.log(1 - y))

    def dloss_dw(self, t, y, x):
        d_loss = y - t
        return x.T.dot(d_loss) / t.shape[0]

    def fit(self, X, y):
        
        for epoch in range(self.epochs):
            for i in range(0, X.shape[0], self.batch_size):
                X_batch = X[i:i+self.batch_size]
                y_batch = y[i:i+self.batch_size]
                y_proba = self.predict_proba(X_batch)
                gr = self.dloss_dw(y_batch, y_proba, X_batch)
                self.wieghts -= self.lr * gr

            y_proba = self.predict_proba(X)
            auc = roc_auc_score(y, y_proba)
            if ((epoch + 1) % 10 == 0 or epoch == 0) and self.verbose:
                print(f"Epoch {epoch+1}/{self.epochs}, AUC: {auc:.4f}")

    def predict(self, X):
        p = self.predict_proba(X)
        y = (p >= 0.5).astype(int)
        return y


    def predict_proba(self, X):
        X = X.astype(np.float32)
        logit = np.dot(X, self.wieghts)
        p = self.sigmiod(logit)
        return p


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a logistic regression model on CSV data")
    parser.add_argument("filename", help="Path to the CSV file")
    parser.add_argument('--save_model', type=int, default=1, help="Flag to save the trained model")

    args = parser.parse_args()
    file_name = args.filename
    save_model = args.save_model

    # load data
    data = load_data(file_name)

    # preprocess data
    data, params_dict = preprocess_data(data)

    # split features and target variable
    X = np.array(data.drop(columns=["Hogwarts House"])).astype(np.float32)
    y = data["Hogwarts House"].values.reshape(-1, 1)

    # split into training and testing sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=70, stratify=y)

    # scale features with standardization
    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_val = scaler.transform(X_val)

    # train logistic regression model
    multi_clsf = OneVsAllClassifier(batch_size=128, lr=0.01, epochs=100)
    clfs_wieghts_df = multi_clsf.fit(X_train, y_train)

    # make predictions
    y_proba = multi_clsf.predict_proba(X_val).reshape(-1, 1)
    y_pred = multi_clsf.predict(X_val).reshape(-1, 1)

    # confusion matrix
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_val, y_pred))
    
    # evaluate model
    accuracy = accuracy_score(y_val, y_pred)
    print(f"\nAccuracy: {accuracy:.4f}")

    # print("Classification Report:")
    # print(classification_report(y_val, y_pred))
    
    y_proba_all = multi_clsf.predict_proba_all(X_val)
    y_val = OneHotEncoder(sparse_output=False).fit_transform(y_val)
    auc = roc_auc_score(y_val, y_proba_all, multi_class='ovr')
    print(f'\nAUC: {auc}')

    # save the trained model
    save_path = 'logreg_params.pkl'
    params_dict['scaler'] = scaler
    if save_model:
        clfs_wieghts_df.to_csv('logreg_weights.csv', index=False)
        with open(save_path, 'wb') as f:
            pkl.dump(params_dict, f)
