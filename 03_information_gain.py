import numpy as np

def entropy(y):
    _, counts = np.unique(y, return_counts=True)
    p = counts / counts.sum()
    return -np.sum(p * np.log2(p))

def information_gain(y, mask):
    parent = entropy(y)
    left, right = y[mask], y[~mask]
    n = len(y)
    ig = parent - (len(left)/n * entropy(left) + len(right)/n * entropy(right))
    return ig

if __name__ == "__main__":
    y = np.array([0, 0, 1, 1])
    mask = np.array([True, False, True, False])
    print("Information Gain:", information_gain(y, mask))
