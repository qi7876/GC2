import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

# 1. generate the sample dataset
np.random.seed(42)
X = np.linspace(0, 2 * np.pi, 1000).reshape(-1, 1)
y = np.sin(X).ravel()

# 2. preprocess for dataset
scaler_X = StandardScaler()
X_scaled = scaler_X.fit_transform(X)

scaler_y = StandardScaler()
y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()

# 3. split the dataset into train set and test set
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_scaled, test_size=0.2, random_state=42
)

print(f"Train Set Size: {X_train.shape}")
print(f"Test Set Size: {X_test.shape}")

# 4. build the BP neural network
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

# 5. train model
mlp.fit(X_train, y_train)

# 6. evaluate the accuracy of model
y_pred_scaled = mlp.predict(X_test)
y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
y_test_actual = scaler_y.inverse_transform(y_test.reshape(-1, 1)).ravel()

mse = mean_squared_error(y_test_actual, y_pred)
print(f"MSE Loss on test set: {mse:.4f}")

# 7. plot the fitting results
# prepare the data for plot
X_test_original = scaler_X.inverse_transform(X_test)
sorted_indices = np.argsort(X_test_original.ravel())
X_test_sorted = X_test_original[sorted_indices].ravel()
y_pred_sorted = y_pred[sorted_indices]
y_test_sorted = y_test_actual[sorted_indices]

# generate a denser set of x to plot real sin wave
x_dense = np.linspace(0, 2 * np.pi, 1000)
y_dense = np.sin(x_dense)

# polt the result
plt.figure(figsize=(12, 6))
plt.plot(x_dense, y_dense, label="Real Sin Wave", color="blue")
plt.scatter(X_test_sorted, y_pred_sorted, label="BP Fit", color="red", s=10)
plt.title("BP Fit Sin Wave")
plt.xlabel("x")
plt.ylabel("sin(x)")
plt.legend()
plt.show()


# 8. input x and predict the output value then calculate the difference between predicted value and real value
def predict_and_compare(x_value):
    # standardize the x
    x_scaled = scaler_X.transform(np.array([[x_value]]))

    # predict the y through model
    y_pred_scaled = mlp.predict(x_scaled)
    y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()[0]

    # calculate the real sin value
    y_real = np.sin(x_value)

    # calculate the difference
    error = abs(y_real - y_pred)

    print(f"\nInput x: {x_value}")
    print(f"Model Predicted y: {y_pred}")
    print(f"Real sin(x): {y_real}")
    print(f"Error: {error}")


# wait for x input
while True:
    user_input = input("\nEnter a value for x (or type 'exit' to quit): ")
    if user_input.lower() == "exit":
        break
    try:
        x_value = float(user_input)
        predict_and_compare(x_value)
    except ValueError:
        print("Invalid input. Please enter a valid number.")
