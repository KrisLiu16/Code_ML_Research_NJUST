import os
import numpy as np
from src.data_loader import load_data
from src.model import time_series_cv
from src.visualization import create_plot, plot_feature_weights
from src.utils import open_html
from src.config import config
from src.train_test_split import train_and_evaluate

if __name__ == "__main__":
    # 数据准备
    max_pre = config['max_pre']
    feature_number = config['feature_number']
    lasso_alphas = config['lasso_alphas']
    ridge_alphas = config['ridge_alphas']
    test_size = config['test_size']
    result_folder = config['result_folder']  # 获取结果文件夹路径

    # 检查并创建结果文件夹（如果不存在）
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)

    X, y = load_data(feature_number)

    # 打印数据
    print("特征 (X):")
    print(X)
    print("目标 (y):")
    print(y)

    # 训练模型并评估
    lasso, y_pred_lasso, y_pred_ridge, y_combined, y_train, metrics, lasso_weight_proportions, ridge_weight_proportions = train_and_evaluate(
        X, y, lasso_alphas, ridge_alphas, test_size
    )

    # 绘制 LASSO 特征权重
    feature_names = [f'特征 {i+1}' for i in range(X.shape[1])]  # 特征名称
    plot_feature_weights(lasso_weight_proportions, feature_names, 'LASSO', result_folder)

    # 绘制 Ridge 特征权重
    plot_feature_weights(ridge_weight_proportions, feature_names, 'Ridge', result_folder)

    # 时间序列预测
    true_values, ts_predictions = time_series_cv(
        X, y, lasso, n_splits=3, test_size=4,
        max_samples=int(len(X) * test_size + max_pre)
    )

    # 创建并显示图表
    create_plot(
        y_combined, y_pred_lasso, y_pred_ridge,
        ts_predictions, y_train, feature_number, X, result_folder
    )

    # 打印 LASSO 模型的预测结果
    print("LASSO 预测结果:")
    print(y_pred_lasso)

    # 打印 Ridge 模型的预测结果
    print("Ridge 预测结果:")
    print(y_pred_ridge)

    # 打印时间序列交叉验证的预测结果
    print("时间序列 LASSO 预测结果:")
    print(ts_predictions)

    # 打开 HTML 文件
    open_html(os.path.join(result_folder, 'regression_predictions.html'))  # 确保打开正确的文件路径
