
import pandas as pd
import numpy as np

np.random.seed(0)


def shuffle(X, y):
    # Shuffle training data
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    return X[indices], y[indices]


def binary_cross_entropy(y_true, y_pred, w):
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))


def binary_cross_entropy_reg(y_true, y_pred, w, alpha=0.01):
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)) + alpha * np.dot(w, w)

def sgd(X_train, y_train, lr=0.001, epochs=1000, batch_size=10000, reg=None):
    # Initialize weights
    w = np.random.normal(0, 1, size=X_train.shape[1])
    b = 0

    # Early stopping
    best_loss = 1e9
    early_stopping_criterion = 1e-5

    loss_fn = binary_cross_entropy_reg if reg == "l2" else binary_cross_entropy

    # Total number of samples
    N = len(y_train)
    for epoch in range(epochs):

        # Shuffle training data
        X_train, y_train = shuffle(X_train, y_train)

        # Mini-batch gradient descent
        for i in range(0, N, batch_size):
            X_train_mini = X_train[i:i+batch_size]
            y_train_mini = y_train[i:i+batch_size]

            weights = class_weights[0] * (1 - y_train_mini) + class_weights[1] * y_train_mini

            # Hypothesis
            y_pred = 1 / (1 + np.exp(-np.dot(X_train_mini, w) - b))

            # Gradient calculation
            w_grad = -(1/N) * X_train_mini.T @ ((y_train_mini - y_pred) * weights)
            b_grad = -(1/N) * np.sum((y_train_mini - y_pred) * weights)

            # Update weights
            w = w - lr * w_grad
            b = b - lr * b_grad

        # Print loss every 100 epochs
        if epoch % 100 == 0:
            y_pred = 1 / (1 + np.exp(-np.dot(X_train, w) - b))
            print(f"Epoch: {epoch}, loss: {loss_fn(y_train, y_pred, w)}")

            # Early stopping
            if loss_fn(y_train, y_pred, w) < best_loss - early_stopping_criterion:
                best_loss = loss_fn(y_train, y_pred, w)
            else:
                print("Stopping early")
                break

    return w, b

def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)


df = pd.read_csv("possessions_18-19.csv")
# print(df.head())
# print(df.describe())

# O reb: this possession
# D reb: next possession
# Steal: next possession
# Block: this possession

df[["Steal"]] = df.groupby(["GameID"]).shift(1)[["Steal"]]
df.dropna(inplace=True)

# Drop all-zero rows
df = df[(df[["Offensive_rebound", "Defensive_rebound", "Steal", "Block"]] != 0).any(axis=1)]

train_size = int(0.8 * len(df))

X_train = df[["Offensive_rebound", "Defensive_rebound", "Steal", "Block"]].iloc[:train_size].copy().to_numpy()
y_train = (df["Points"].iloc[:train_size] > 0).copy().to_numpy().astype(int)
X_test = df[["Offensive_rebound", "Defensive_rebound", "Steal", "Block"]].iloc[train_size:].copy().to_numpy()
y_test = (df["Points"].iloc[train_size:] > 0).copy().to_numpy().astype(int)

# class_weights = np.array([1 / np.sum(y_train == 0), 1 / np.sum(y_train == 1)])
class_weights = np.array([1, 1.03])

w, b = sgd(X_train, y_train, lr=0.1, epochs=10000, batch_size=10000)
y_pred = 1 / (1 + np.exp(-np.dot(X_test, w) - b))
print(f"Test loss: {binary_cross_entropy(y_test, y_pred, w)}")
y_pred = (y_pred > 0.5).astype(int)
print(f"Test accuracy: {accuracy(y_test, y_pred)}")

breakpoint()
