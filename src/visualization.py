import plotly.graph_objs as go
import plotly.offline as pyo
import matplotlib.pyplot as plt
import numpy as np
import os
def rgb_to_hex(rgb):
    return '#%02x%02x%02x' % (int(rgb[0]*255), int(rgb[1]*255), int(rgb[2]*255))

def create_plot(y_combined, y_pred_lasso, y_pred_ridge, ts_predictions, y_train, feature_number, X, result_folder):
    # 创建图表数据
    actual_trace = go.Scatter(
        x=list(range(len(y_combined))),
        y=y_combined,
        mode='lines',
        name='实际目标值',
        line=dict(color='blue')
    )

    lasso_trace = go.Scatter(
        x=list(range(len(y_combined))),
        y=y_pred_lasso,
        mode='lines',
        name='LASSO 预测值',
        line=dict(color='red')
    )

    ridge_trace = go.Scatter(
        x=list(range(len(y_combined))),
        y=y_pred_ridge,
        mode='lines',
        name='Ridge 预测值',
        line=dict(color='green')
    )

    # 添加原始特征数据的图表数据
    feature_traces = []
    colors = plt.cm.viridis(np.linspace(0, 1, feature_number))  # 使用 Viridis 颜色映射

    for i in range(feature_number):
        feature_data = X[:, i]
        feature_trace = go.Scatter(
            x=list(range(len(feature_data))),
            y=feature_data,
            mode='lines',
            name=f'特征 {i+1}',
            line=dict(color=rgb_to_hex(colors[i])),  # 使用 Hex 颜色代码
            visible='legendonly'
        )
        feature_traces.append(feature_trace)

    # 确保时间序列预测的 x 轴范围正确
    ts_lasso_trace = go.Scatter(
        x=list(range(len(y_train), len(y_train) + len(ts_predictions))),
        y=ts_predictions,
        mode='lines',
        name='时间序列 LASSO 预测值',
        line=dict(color='orange'),
        visible='legendonly'  # 默认不可见
    )

    boundary_line = go.Scatter(
        x=[len(y_train)-1, len(y_train)-1],
        y=[min(y_combined), max(y_combined)],
        mode='lines',
        name='训练/测试边界',
        line=dict(color='black', dash='dash')
    )

    # 合并所有数据
    data = [actual_trace, lasso_trace, ridge_trace, ts_lasso_trace, boundary_line] + feature_traces

    # 设置图表布局
    layout = go.Layout(
        title='回归预测与特征数据',
        xaxis=dict(title='样本'),
        yaxis=dict(title='值')
    )

    # 创建并保存图表
    fig = go.Figure(data=data, layout=layout)
    result_file = os.path.join(result_folder, 'regression_predictions.html')  # 拼接文件路径
    pyo.plot(fig, filename=result_file, auto_open=False)

def plot_feature_weights(weights, feature_names, model_name, result_folder):
    # 创建特征权重条形图
    trace = go.Bar(
        x=feature_names,
        y=weights,
        marker=dict(color='skyblue'),
        name=f'{model_name} 特征权重'
    )

    layout = go.Layout(
        title=f'{model_name} 特征权重',
        xaxis=dict(title='特征'),
        yaxis=dict(title='权重占比 (%)'),
        showlegend=False
    )

    fig = go.Figure(data=[trace], layout=layout)
    result_file = os.path.join(result_folder, f'{model_name}_feature_weights.html')  # 拼接文件路径
    pyo.plot(fig, filename=result_file, auto_open=False)