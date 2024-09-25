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
plt.scatter(X, y, color="lightgray", label="Real Data")
plt.title("Sin Wave Data Distibution")
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

print(f"Train Set Size: {X_train.shape}")
print(f"Test Set Size: {X_test.shape}")

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
print(f"MSE Loss on test set: {mse:.4f}")

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
plt.plot(x_dense, y_dense, label="Real Sin Wave", color="blue")
plt.scatter(X_test_sorted, y_pred_sorted, label="BP Fit", color="red", s=10)
plt.title("BP Fit Sin Wave")
plt.xlabel("x")
plt.ylabel("sin(x)")
plt.legend()
plt.show()


# 8. 输入 x 值并输出模型预测值和误差
def predict_and_compare(x_value):
    # 将 x_value 进行标准化
    x_scaled = scaler_X.transform(np.array([[x_value]]))

    # 通过模型预测 y 值
    y_pred_scaled = mlp.predict(x_scaled)
    y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()[0]

    # 计算真实的正弦函数值
    y_real = np.sin(x_value)

    # 计算误差
    error = abs(y_real - y_pred)

    print(f"\nInput x: {x_value}")
    print(f"Model Predicted y: {y_pred}")
    print(f"Real sin(x): {y_real}")
    print(f"Error: {error}")


# 等待用户输入 x 值
while True:
    user_input = input("\nEnter a value for x (or type 'exit' to quit): ")
    if user_input.lower() == "exit":
        break
    try:
        x_value = float(user_input)
        predict_and_compare(x_value)
    except ValueError:
        print("Invalid input. Please enter a valid number.")
