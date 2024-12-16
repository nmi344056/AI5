import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def silu(x):                            # s = sigmoid
    return x * (1 / (1 + np.exp(-x)))   # x * sigmoid

# silu = lambda x : x * (1 / (1 + np.exp(-x)))

x = np.arange(-5, 5, 0.1)
y = silu(x)

plt.plot(x, y)
plt.grid()
plt.show()

# 문제점 : ReLu 보다 계산량이 많아서 모델이 커질수록 부담스럽니다.
