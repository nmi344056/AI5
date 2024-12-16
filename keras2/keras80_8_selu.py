import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def selu(x):
    alpha = 1.6733
    lambda_ = 1.0507
    return lambda_ * np.where(x >= 0, x, alpha * (np.exp(x) - 1))

# selu = lambda x : lambda_ * np.where(x >= 0, x, alpha * (np.exp(x) - 1))

x = np.arange(-5, 5, 0.1)
y = selu(x)

# def selu(x, alpha, scale):
#     return np.where (x <= 0, scale * alpha * (np.exp(x)-1), scale * x)

# x = np.arange(-5, 5, 0.1)
# alpha = 1
# scale = 1
# y = selu(x, alpha, scale)

plt.plot(x, y)
plt.grid()
plt.show()



