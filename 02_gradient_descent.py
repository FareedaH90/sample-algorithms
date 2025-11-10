import numpy as np

# y = 3x + 2 (ground truth)
X = np.array([1, 2, 3, 4, 5])
y = 3 * X + 2

m, b = 0, 0
lr = 0.01

for _ in range(1000):
    y_pred = m * X + b
    dm = (-2 / len(X)) * np.sum(X * (y - y_pred))
    db = (-2 / len(X)) * np.sum(y - y_pred)
    m -= lr * dm
    b -= lr * db

print(f"Learned parameters -> m: {round(m, 2)}, b: {round(b, 2)}")