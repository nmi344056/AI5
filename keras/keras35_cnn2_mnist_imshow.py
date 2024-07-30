import numpy as np
from tensorflow.keras.datasets import mnist
import pandas as pd

(x_train, y_train), (x_test, y_test) = mnist.load_data()

np.set_printoptions(edgeitems=30, linewidth = 1024)

# print(x_train)   0 이 많이 달리는이유 = 그림 가운데 숫자가 몰려있어서 앞뒤에 0 이 많다.
print(x_train[0])
print("y_train[0] : ", y_train[0])

print(x_train.shape, y_train.shape)     # (60000, 28, 28) (60000,)   흑백데이터라 맨뒤에 1이 생략 -- 변환시켜준다
print(x_test.shape, y_test.shape)       # (10000, 28, 28) (10000,)

print(np.unique(y_train, return_counts=True))      # y 값 확인
# (array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8), array([5923, 6742, 5958, 6131, 5842, 5421, 5918, 6265, 5851, 5949], dtype=int64))
print(pd.value_counts(y_test))

import matplotlib.pyplot as plt
plt.imshow(x_train[8644], 'winter')
plt.show()