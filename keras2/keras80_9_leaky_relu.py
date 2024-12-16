import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def leaky_relu(x, alpha=0.1):
    return np.where(x >= 0, x, alpha * x)

# def leaky_relu(x):
#     return np.maximum(0.01 * x, x) 

# leaky_relu = lambda x : np.where(x >= 0, x, alpha * x)

x = np.arange(-5, 5, 0.1)
y = leaky_relu(x)

plt.plot(x, y)
plt.grid()
plt.show()
