import numpy as np

def mae(t, pred):
    return np.mean(np.abs(pred - t))

def mse(t, pred):
    return np.mean((pred - t) ** 2)
