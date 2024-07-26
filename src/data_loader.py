import pandas as pd
import numpy as np

def load_data(feature_number):
    X = []
    for i in range(1, feature_number + 1):
        data = pd.read_csv(f'data/feature{i}.csv', header=None)
        data = data.iloc[::-1].reset_index(drop=True)  # 倒序读取
        X.append(data.values)
    X = np.hstack(X)
    y = pd.read_csv('data/target.csv', header=None)
    y = y.iloc[::-1].reset_index(drop=True).values.ravel()  # 倒序读取
    return X, y
