import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# 1. 加载数据集
digits = load_digits()
X = digits.data
y = digits.target

print(f"数据集大小: {X.shape}")
print(f"标签类别: {np.unique(y)}")

# 2. 数据预处理
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

print(f"训练集大小: {X_train.shape}")
print(f"测试集大小: {X_test.shape}")

# 4. 构建BP神经网络模型
mlp = MLPClassifier(
    hidden_layer_sizes=(100,),
    activation="relu",
    solver="adam",
    alpha=1e-4,
    batch_size="auto",
    learning_rate="constant",
    learning_rate_init=0.001,
    max_iter=200,
    random_state=42,
    early_stopping=True,
    n_iter_no_change=10,
)

# 5. 训练模型
mlp.fit(X_train, y_train)

# 6. 评估模型性能
y_pred = mlp.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"模型准确率: {accuracy * 100:.2f}%")

print("分类报告:")
print(classification_report(y_test, y_pred))

print("混淆矩阵:")
print(confusion_matrix(y_test, y_pred))

# 7. 预测和可视化结果
num_samples = 10
indices = np.random.choice(len(X_test), num_samples, replace=False)

plt.figure(figsize=(10, 5))
for i, idx in enumerate(indices):
    image = X_test[idx].reshape(8, 8)
    true_label = y_test[idx]
    pred_label = y_pred[idx]

    plt.subplot(2, 5, i + 1)
    plt.imshow(image, cmap="gray")
    plt.title(f"真实: {true_label}\n预测: {pred_label}")
    plt.axis("off")

plt.tight_layout()
plt.show()
