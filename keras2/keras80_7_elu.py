import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def elu(x, alpha=1.0):
    return np.where(x >= 0, x, alpha * (np.exp(x) - 1))

# elu = lambda x : np.where(x >= 0, x, alpha * (np.exp(x) - 1))

x = np.arange(-5, 5, 0.1)
y = elu(x)

plt.plot(x, y)
plt.grid()
plt.show()

#####
# def elu(x, alp):
#     return (x>0)*x + (x<=0)*(alp + (np.exp(x) -1))  

# x = np.arange(-5, 5, 0.1)
# y = elu(x, alp= 0.1)
