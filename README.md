# 回归预测与特征数据分析

该项目用于使用LASSO和Ridge回归模型对时间序列数据进行预测，并可视化实际值与预测值。项目结构模块化，方便维护和扩展。

## 项目结构

```
Code_ML/
│
├── data/                   # 数据存放处
│
├── src/                    # 存放代码的文件夹
│   ├── __init__.py         # 使 src 成为一个 Python 包
│   ├── data_loader.py      # 数据加载模块
│   ├── model.py            # 模型训练与预测模块
│   ├── visualization.py    # 数据可视化模块
│   ├── config.py           # 参数设置
│   ├── train_test_split.py # 数据规则化模块
│   └── utils.py            # 辅助函数模块
│
├── main.py                 # 主脚本，负责运行主要流程
├── README.md
└── regression_predictions.html
```

## 依赖安装

在运行项目之前，请确保已安装所有必需的Python库。您可以使用以下命令安装依赖项：

```sh
pip install numpy pandas plotly scikit-learn matplotlib
```

## 数据准备

确保您的数据文件位于 `data` 目录中。项目假定数据文件名为 `feature1.csv`, `feature2.csv`, ..., `featureN.csv` 和 `target.csv`，其中 `N` 是特征数量。

数据文件格式：

- `featureX.csv`：每个文件包含一个特征的数据。
- `target.csv`：包含目标值的数据。

数据应按时间顺序存储，项目中将其倒序读取。

## 使用说明

1. **配置参数：**

   在 `main.py` 中，可以根据需要配置以下参数：
   
   - `max_pre`：最大预测步数。
   - `feature_number`：特征数量。
   - `lasso_alphas`：LASSO模型的正则化参数。
   - `ridge_alphas`：Ridge模型的正则化参数。
   - `test_size`：测试集比例。

2. **运行项目：**

   运行 `main.py` 来执行主要流程：

   ```sh
   python main.py
   ```

   该脚本将加载数据，训练模型，进行预测，并生成可视化图表。

3. **查看结果：**

   预测结果将保存在 `regression_predictions.html` 文件中，并自动在默认浏览器中打开。

## 模块说明

### `main.py`

主脚本，负责数据加载、模型训练、预测和结果可视化。主要步骤包括：

- 加载数据。
- 拆分训练集和测试集。
- 标准化数据。
- 训练LASSO和Ridge模型。
- 进行时间序列交叉验证预测。
- 生成并保存可视化图表。

### `data_loader.py`

包含数据加载函数：

```python
def load_data(feature_number):
    # 加载特征和目标数据
```

### `model.py`

包含模型训练和预测函数：

```python
def train_lasso(X_train, y_train, alphas):
    # 训练LASSO模型

def train_ridge(X_train, y_train, alphas):
    # 训练Ridge模型

def time_series_cv(X, y, model, n_splits, test_size, max_samples):
    # 时间序列交叉验证和预测
```

### `visualization.py`

包含数据可视化函数：

```python
def create_plot(y_combined, y_pred_lasso, y_pred_ridge, ts_predictions, y_train, feature_number, X):
    # 创建并保存可视化图表
```

### `utils.py`

包含辅助函数：

```python
def open_html(filename):
    # 打开HTML文件
```

## 注意事项

- 请确保数据文件存在并格式正确。
- 修改参数时，请确保与数据文件中的特征数量匹配。