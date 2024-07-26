import numpy as np
from sklearn.linear_model import LassoCV, RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit

def train_lasso(X_train, y_train, alphas):
    lasso = LassoCV(alphas=alphas, cv=5, random_state=42).fit(X_train, y_train)
    return lasso

def train_ridge(X_train, y_train, alphas):
    ridge = RidgeCV(alphas=alphas, store_cv_values=True).fit(X_train, y_train)
    return ridge

def time_series_cv(X, y, model, n_splits=3, test_size=4, max_samples=100):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    predictions = []
    true_values = []
    for train_index, test_index in tscv.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # 标准化
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        predictions.extend(y_pred)
        true_values.extend(y_test)

        # 截断到指定样本数量
        if len(predictions) >= max_samples:
            break

    # 确保 predictions 和 true_values 不超过 max_samples
    return np.array(true_values[:max_samples]), np.array(predictions[:max_samples])
