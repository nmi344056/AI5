# https://www.kaggle.com/competitions/dogs-vs-cats-redux-kernels-edition/overview

# 42_1 복사 (#1. 데이터만)

import numpy as np
import pandas as pd
import time
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score

#1. 데이터

start1 = time.time()

np_path = 'C:\\ai5\\_data\\_save_npy\\'
# np.save(np_path + 'keras43_01_x_train.npy', arr=xy_train[0][0])
# np.save(np_path + 'keras43_01_y_train.npy', arr=xy_train[0][1])
# np.save(np_path + 'keras43_01_x_test.npy', arr=xy_test[0][0])
# np.save(np_path + 'keras43_01_y_test.npy', arr=xy_test[0][1])

x_train = np.load(np_path + 'keras43_01_x_train.npy')
y_train = np.load(np_path + 'keras43_01_y_train.npy')
x_test = np.load(np_path + 'keras43_01_x_test.npy')
y_test = np.load(np_path + 'keras43_01_y_test.npy')

# print(x_train)
print(x_train.shape)                            # (25000, 100, 100, 3)
# print(y_train)
print(y_train.shape)                            # (25000,)
# print(x_test)
print(x_test.shape)                             # (12500, 100, 100, 3)

print(y_test)                                   # [0. 0. 0. ... 0. 0. 0.] y_test는 할 필요 없다
print(y_test.shape)                             # (12500,)

end1 = time.time()
print("time :", round(end1 - start1, 2),'초')   # time : 0.98 초
