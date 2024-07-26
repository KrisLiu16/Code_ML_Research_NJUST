import numpy as np
from sklearn.preprocessing import StandardScaler
from .model import train_lasso, train_ridge

def train_and_evaluate(x, y, lasso_alphas, ridge_alphas, test_size=0.1):
    # 手动拆分训练集和测试集
    split_index = int(len(x) * (1 - test_size))
    x_train, x_test = x[:split_index], x[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]

    # 标准化数据
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    # 训练LASSO模型
    lasso = train_lasso(x_train_scaled, y_train, lasso_alphas)
    y_pred_lasso_train = lasso.predict(x_train_scaled)
    y_pred_lasso_test = lasso.predict(x_test_scaled)

    # 训练Ridge回归模型
    ridge = train_ridge(x_train_scaled, y_train, ridge_alphas)
    y_pred_ridge_train = ridge.predict(x_train_scaled)
    y_pred_ridge_test = ridge.predict(x_test_scaled)

    # 合并训练集和测试集的预测结果
    y_pred_lasso = np.concatenate((y_pred_lasso_train, y_pred_lasso_test))
    y_pred_ridge = np.concatenate((y_pred_ridge_train, y_pred_ridge_test))

    # 合并实际值
    y_combined = np.concatenate((y_train, y_test))

    return lasso, y_pred_lasso, y_pred_ridge, y_combined, y_train
