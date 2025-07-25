import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, OneHotEncoder
from sklearn.datasets import load_iris

# بارگذاری دیتای چندکلاسه (Iris)
data = load_iris()
x = data["data"].T  # shape: (features, samples)
y = data["target"]

# One-Hot Encoding برچسب‌ها
encoder = OneHotEncoder(sparse=False)
y = encoder.fit_transform(y.reshape(-1, 1)).T  # shape: (classes, samples)

# نرمال‌سازی ویژگی‌ها
scaler = MinMaxScaler()
x = scaler.fit_transform(x.T).T

# تقسیم دیتا
x_train, x_test, y_train, y_test = train_test_split(x.T, y.T, test_size=0.2, random_state=4)
x_train = x_train.T
y_train = y_train.T
x_test = x_test.T
y_test = y_test.T

print("x_train shape:", x_train.shape)
print("y_train shape:", y_train.shape)
print("x_test shape:", x_test.shape)
print("y_test shape:", y_test.shape)

def initialize(n_0, n_1):
    w_i = np.random.randn(n_1, n_0) * np.sqrt(2 / n_0)
    b_i = np.random.randn(n_1, 1) * 0.01
    return (w_i, b_i)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def relu(z):
    return np.maximum(0, z)

def relu_prime(z):
    return (z > 0).astype(float)

def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=0, keepdims=True))  # پایداری عددی
    return exp_z / np.sum(exp_z, axis=0, keepdims=True)

def forward(x, y, cash, j, l):
    n_o = 0
    if j == 0:
        a = x
        cash["a_0"] = a
        for i in range(l):
            n = int(input("How many nodes does the layer " + str(i + 1) + " have?  "))
            if i == 0:
                w_i, b_i = initialize(x.shape[0], n)
            else:
                w_i, b_i = initialize(n_o, n)

            z = np.dot(w_i, a) + b_i
            a = softmax(z) if i == l - 1 else relu(z)

            cash["z_" + str(i + 1)] = z
            cash["a_" + str(i + 1)] = a
            cash["w_" + str(i + 1)] = w_i
            cash["b_" + str(i + 1)] = b_i

            n_o = n
    else:
        a = x
        cash["a_0"] = a
        for i in range(l):
            w_i = cash["w_" + str(i + 1)]
            b_i = cash["b_" + str(i + 1)]
            z = np.dot(w_i, a) + b_i
            a = softmax(z) if i == l - 1 else relu(z)

            cash["z_" + str(i + 1)] = z
            cash["a_" + str(i + 1)] = a

    y_hat = a
    return cash, y_hat, l

def backward(x, y, cash, y_hat, l, lr):
    m = x.shape[1]
    dz = y_hat - y
    for i in reversed(range(1, l + 1)):
        a_prev = x if i == 1 else cash["a_" + str(i - 1)]
        dw = (1 / m) * np.dot(dz, a_prev.T)
        db = (1 / m) * np.sum(dz, axis=1, keepdims=True)

        cash["w_" + str(i)] -= lr * dw
        cash["b_" + str(i)] -= lr * db

        if i > 1:
            dz = np.dot(cash["w_" + str(i)].T, dz) * relu_prime(cash["z_" + str(i - 1)])

    cost = - (1 / m) * np.sum(y * np.log(y_hat + 1e-8))
    return cash, cost

def optimize(x, y, l, lr, ni, print_cost=False):
    costs = []
    cash = dict()
    for j in range(ni):
        cash, y_hat, l = forward(x, y, cash, j, l)
        cash, cost = backward(x_train, y_train, cash, y_hat, l, lr)
        if j % 100 == 0:
            costs.append(cost)
            if print_cost:
                print(f"Cost after iteration {j}: {cost:.6f}")
    return cash, costs

def predict(x, y, cash, l):
    a = x
    for i in range(1, l + 1):
        z = np.dot(cash["w_" + str(i)], a) + cash["b_" + str(i)]
        a = softmax(z) if i == l else relu(z)
    return np.argmax(a, axis=0)

def model(x_train, y_train, x_test, y_test, l, learning_rate=0.005, num_iterations=1000, print_cost=True):
    cash, _ = optimize(x_train, y_train, l, learning_rate, num_iterations, print_cost)
    y_predict = predict(x_test, y_test, cash, l)
    accuracy = np.mean(y_predict == np.argmax(y_test, axis=0)) * 100
    return round(accuracy, 2)

# اجرا
l = int(input("How many layers does the model have?  "))
accuracy = model(x_train, y_train, x_test, y_test, l, learning_rate=0.01, num_iterations=10000, print_cost=True)
print(f"Test set accuracy: {accuracy}%")
