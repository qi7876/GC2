import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

# 1. 生成样本数据
np.random.seed(42)  # 设置随机种子以保证结果可重复
X = np.linspace(0, 2 * np.pi, 1000).reshape(-1, 1)  # 1000个样本点
y = np.sin(X).ravel()  # 正弦函数值

# 可视化数据
plt.figure(figsize=(10, 4))
plt.scatter(X, y, color="lightgray", label="真实数据")
plt.title("正弦函数数据分布")
plt.xlabel("x")
plt.ylabel("sin(x)")
plt.legend()
plt.show()

# 2. 数据预处理
scaler_X = StandardScaler()
X_scaled = scaler_X.fit_transform(X)

scaler_y = StandardScaler()
y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()

# 3. 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_scaled, test_size=0.2, random_state=42
)

print(f"训练集大小: {X_train.shape}")
print(f"测试集大小: {X_test.shape}")

# 4. 构建BP神经网络模型
mlp = MLPRegressor(
    hidden_layer_sizes=(100, 100),
    activation="tanh",
    solver="adam",
    alpha=1e-4,
    learning_rate="adaptive",
    max_iter=10000,
    tol=1e-6,
    random_state=42,
)

# 5. 训练模型
mlp.fit(X_train, y_train)

# 6. 评估模型性能
y_pred_scaled = mlp.predict(X_test)
y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
y_test_actual = scaler_y.inverse_transform(y_test.reshape(-1, 1)).ravel()

mse = mean_squared_error(y_test_actual, y_pred)
print(f"模型在测试集上的均方误差: {mse:.4f}")

# 7. 绘制拟合结果
# 准备绘图数据
X_test_original = scaler_X.inverse_transform(X_test)
sorted_indices = np.argsort(X_test_original.ravel())
X_test_sorted = X_test_original[sorted_indices].ravel()
y_pred_sorted = y_pred[sorted_indices]
y_test_sorted = y_test_actual[sorted_indices]

# 生成更密集的x用于绘制真实的正弦曲线
x_dense = np.linspace(0, 2 * np.pi, 1000)
y_dense = np.sin(x_dense)

# 绘制结果
plt.figure(figsize=(12, 6))
plt.plot(x_dense, y_dense, label="真实正弦曲线", color="blue")
plt.scatter(X_test_sorted, y_pred_sorted, label="BP神经网络拟合", color="red", s=10)
plt.title("BP神经网络拟合正弦曲线")
plt.xlabel("x")
plt.ylabel("sin(x)")
plt.legend()
plt.show()
