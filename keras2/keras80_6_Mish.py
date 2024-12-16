import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# def mish(x):
#     return x * np.tanh(np.log(1 + np.exp(x)))

mish = lambda x : x * np.tanh(np.log(1 + np.exp(x)))

x = np.arange(-5, 5, 0.1)
y = mish(x)

plt.plot(x, y)
plt.grid()
plt.show()

# 연산량이 훨씬 많다.

# [실습] elu, selu, meaky_relu
