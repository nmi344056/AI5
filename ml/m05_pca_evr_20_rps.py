# 37_1 copy

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM, GRU
from tensorflow.keras.layers import Conv1D, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score

#1. 데이터
np_path = 'C:\\ai5\\_data\\_save_npy\\keras45\\'
x_train = np.load(np_path + 'keras45_03_rps_x_train.npy')
y_train = np.load(np_path + 'keras45_03_rps_y_train.npy')

x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, train_size=0.8, random_state=123)

print(x_train.shape, y_train.shape) # (24, 100, 100, 3) (24, 3)
print(x_test.shape, y_test.shape)   # (6, 100, 100, 3) (6, 3)

train_datagen = ImageDataGenerator(
    # rescale=1./255,     # 처음부터 0~1 사이의 스케일링한 데이터를 달라
    horizontal_flip=True,   # 수평 뒤집기
    vertical_flip=True,     # 수직 뒤집기
    width_shift_range=0.2,  # 평행 이동, 10%만큼 옆으로 이동
    # height_shift_range=0.1, # 평행 이동 수직
    rotation_range=15,       # 정해진 각도만큼 이미지 회전
    # zoom_range=1.2,         # 축소 또는 확대
    # shear_range=0.7,        # 좌표 하나를 고정시키고 다른 몇개의 좌표를 이동시키는 변환 (일그러짐)
    fill_mode='nearest',    # 주변의 이미지의 비율대로 채운다
)

augment_size = 10000

randidx = np.random.randint(x_train.shape[0], size=augment_size)
# print(randidx)
# print(np.min(randidx), np.max(randidx))
print(x_train[0].shape)     # (28, 28)--

x_augmented = x_train[randidx].copy()       # .copy() 하면 메모리공간 새로 할당, 영향을 미치지 않는다, 비파괴적
y_augmented = y_train[randidx].copy()       # x, y 순서 주의
print(x_augmented.shape, y_augmented.shape) # (40000, 28, 28) (40000,)

# x_augmented = x_augmented.reshape(40000, 28, 28, 1)     # (40000, 28, 28, 1)으로작성하면 바꿔야한다
# x_augmented = x_augmented.reshape(
#     x_augmented.shape[0],   # 40000
#     x_augmented.shape[1],   # 28
#     x_augmented.shape[2], 1)   # 28

print(x_augmented.shape)    # (40000, 28, 28, 1)

x_augmented = train_datagen.flow(
    x_augmented, y_augmented,
    batch_size=augment_size,
    shuffle=False,
).next()[0]

print(x_augmented.shape)     # (40000, 28, 28, 1)

# x_train = x_train.reshape(60000, 28, 28, 1)
# x_test = x_test.reshape(10000, 28, 28, 1)
# print(x_train.shape, x_test.shape)      # (60000, 28, 28, 1) (10000, 28, 28, 1)

# # [검색] 넘파이 행렬 데이터 합치기
# x_train = np.concatenate((x_train, x_augmented))
# y_train = np.concatenate((y_train, y_augmented))
# print(x_train.shape, y_train.shape)        # (100000, 28, 28, 1) (100000,)





