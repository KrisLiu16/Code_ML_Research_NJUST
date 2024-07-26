import numpy as np

# 配置文件，包含所有参数设置
config = {
    "max_pre": 6,
    "feature_number": 3,
    "lasso_alphas": np.logspace(-5, 5, 200),
    "ridge_alphas": np.logspace(-5, 5, 200),
    "test_size": 0.1
}
