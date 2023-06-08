import numpy as np
import pandas as pd
import pickle

subsample = 0.1
lr = 1e-6
epochs = 10000
batch_size = 10000
np.random.seed(0)


def shuffle(X, y):
    # Shuffle training data
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    return X[indices], y[indices]


def mse_loss(y_true, y_pred):
    return ((y_true - y_pred) ** 2).mean()


def sgd(X_train, y_train, X_val, y_val, lr, epochs, batch_size):
    # Initialize weights
    w = np.random.normal(0, 1, size=X_train.shape[1])
    b = 0

    # Early stopping
    best_loss = 1e9
    early_stopping_criterion = 1e-4

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
            w_grad = -(2 / len(X_train_mini)) * np.dot(X_train_mini.T, (y_train_mini - y_pred))
            b_grad = -(2 / len(X_train_mini)) * np.sum(y_train_mini - y_pred)

            # Update weights
            w = w - lr * w_grad
            b = b - lr * b_grad

        # Print loss every 100 epochs
        if epoch % 100 == 0:
            y_pred_train = np.dot(X_train, w) + b
            y_pred_val = np.dot(X_val, w) + b
            train_loss = mse_loss(y_train, y_pred_train)
            val_loss = mse_loss(y_val, y_pred_val)
            print("Epoch: {}, train loss: {}, val loss {}".format(
                epoch,
                train_loss,
                val_loss
            ), end='\r')

            # Early stopping
            if val_loss < best_loss - early_stopping_criterion:
                best_loss = val_loss
            else:
                print("")
                print("Stopping early")
                break

    return w, b


def r_squared(y_true, y_pred):
    numerator = ((y_true - y_pred) ** 2).sum()
    denominator = ((y_true - y_true.mean()) ** 2).sum()
    return 1 - numerator / denominator


df = pd.read_csv('pbp_with_rolling_stats.csv')
df.head()

# Shift steal col
df[["Steal"]] = df.groupby(["GameID"]).shift(1)[["Steal"]]
df.dropna(inplace=True)

# Subsample 10% of the data
df = df[:int(subsample * len(df))]

df_train = df[:int(len(df) * 0.8)]
df_test = df[int(len(df) * 0.8):]
len(df)

train_cols = ['Time', 'Offensive_rebound', 'Defensive_rebound', 'Steal', 'Block',
              'OffPPM', 'DefPPM', 'OffAPM', 'OffRPM', 'DefAPM', 'DefRPM',
              'OffRaptorOff', 'OffRaptorDef', 'OffRaptorTot', 'OffRaptorWar',
              'DefRaptorOff', 'DefRaptorDef', 'DefRaptorTot', 'DefRaptorWar'
              ]
target_cols = ['Points']

X_train = df_train[train_cols].to_numpy()
y_train = df_train[target_cols].to_numpy().squeeze(-1)
X_test = df_test[train_cols].to_numpy()
y_test = df_test[target_cols].to_numpy().squeeze(-1)

w, b = sgd(X_train, y_train, X_test, y_test,
           lr=lr, epochs=epochs, batch_size=batch_size)

results = {
    'w': w,
    'b': b,
    'train_cols': train_cols,
    'target_cols': target_cols,
    'subsample': subsample,
    'lr': lr,
    'epochs': epochs,
    'batch_size': batch_size,
    'train_loss': mse_loss(y_train, np.dot(X_train, w) + b),
    'test_loss': mse_loss(y_test, np.dot(X_test, w) + b),
    'R2_train': r_squared(y_train, np.dot(X_train, w) + b),
    'R2_test': r_squared(y_test, np.dot(X_test, w) + b),
}

# Save results
with open('ep_model_results.pkl', 'wb') as f:
    pickle.dump(results, f)
