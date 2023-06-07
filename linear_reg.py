
import pandas as pd
import numpy as np

np.random.seed(0)


def shuffle(X, y):
    # Shuffle training data
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    return X[indices], y[indices]


def mse_loss(y_true, y_pred, w):
    return ((y_true - y_pred) ** 2).mean()


def l2_mse_loss(y_true, y_pred, w, alpha=0.01):
    return ((y_true - y_pred) ** 2).mean() + alpha * np.dot(w, w)


def sgd(X_train, y_train, lr=0.001, epochs=1000, batch_size=10000, reg=None):
    # Initialize weights
    w = np.random.normal(0, 1, size=X_train.shape[1])
    b = 0

    # Early stopping
    best_loss = 1e9
    early_stopping_criterion = 1e-6

    loss_fn = l2_mse_loss if reg == "l2" else mse_loss

    # Total number of samples
    N = len(y_train)
    for epoch in range(epochs):

        # Shuffle training data
        X_train, y_train = shuffle(X_train, y_train)

        # Mini-batch gradient descent
        for i in range(0, N, batch_size):
            X_train_mini = X_train[i:i+batch_size]
            y_train_mini = y_train[i:i+batch_size]

            # Hypothesis
            y_pred = np.dot(X_train_mini, w) + b

            # Gradient calculation
            w_grad = -(2/N) * np.dot(X_train_mini.T, (y_train_mini - y_pred))
            b_grad = -(2/N) * np.sum(y_train_mini - y_pred)

            # Update weights
            w = w - lr * w_grad
            b = b - lr * b_grad

        # Print loss every 100 epochs
        if epoch % 100 == 0:
            y_pred = np.dot(X_train, w) + b
            print(f"Epoch: {epoch}, loss: {loss_fn(y_train, y_pred, w)}")

            # Early stopping
            if loss_fn(y_train, y_pred, w) < best_loss - early_stopping_criterion:
                best_loss = loss_fn(y_train, y_pred, w)
            else:
                print("Stopping early")
                break

    return w, b


def r_squared(y_true, y_pred):
    numerator = ((y_true - y_pred) ** 2).sum()
    denominator = ((y_true - y_true.mean()) ** 2).sum()
    return 1 - numerator / denominator


df = pd.read_csv("possessions_18-19.csv")

df[["Steal"]] = df.groupby(["GameID"]).shift(1)[["Steal"]]
df.dropna(inplace=True)

# Drop all-zero rows
df = df[(df[["Offensive_rebound", "Defensive_rebound", "Steal", "Block"]] != 0).any(axis=1)]

train_size = int(0.8 * len(df))

feats = ["Offensive_rebound", "Defensive_rebound", "Steal", "Block"]
X_train = df[feats].iloc[: train_size].to_numpy()
y_train = df["Points"].iloc[: train_size ].to_numpy()
X_test = df[feats].iloc[ train_size :].to_numpy()
y_test = df["Points"].iloc[ train_size :].to_numpy()

print(X_train.shape, y_train.shape)

w, b = sgd(X_train, y_train, lr=0.1, epochs=1000, batch_size=10000)
y_pred = np.dot(X_test, w) + b
print(f"Test loss: {mse_loss(y_test, y_pred, w)}")
print(f"Test R^2: {r_squared(y_test, y_pred)}")

breakpoint()
