# 37_1 copy

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, Input, MaxPooling2D, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score

#1. 데이터
path_np = 'C:\\ai5\\_data\\_save_npy\\keras45\\'
# x_man = np.load(path_np + 'keras45_07_gender_x_man.npy')
# y_man = np.load(path_np + 'keras45_07_gender_y_man.npy')
# x_woman = np.load(path_np + 'keras45_07_gender_x_woman.npy')
# y_woman = np.load(path_np + 'keras45_07_gender_y_woman.npy')

x_train = np.load(path_np + 'keras45_07_gender2_x_train.npy')
y_train = np.load(path_np + 'keras45_07_gender2_y_train.npy')

print(x_train.shape, y_train.shape)     # (27167, 100, 100, 3) (27167,)

x_man = x_train[np.where(y_train <= 0.0)]     # 조건을 만족하는 y값이 있는 인덱스 추출
y_man = y_train[np.where(y_train <= 0.0)]
x_woman = x_train[np.where(y_train > 0.0)]
y_woman = y_train[np.where(y_train > 0.0)]

print(x_man.shape, y_man.shape)         # (17678, 100, 100, 3) (17678,)
print(x_woman.shape, y_woman.shape)     # (9489, 100, 100, 3) (9489,)

x_train_man, x_test_man, y_train_man, y_test_man = train_test_split(x_man, y_man, train_size=0.8, random_state=123)
print(x_train_man.shape, y_train_man.shape)     # (14142, 100, 100, 3) (14142,)
print(x_test_man.shape, y_test_man.shape)       # (3536, 100, 100, 3) (3536,)

x_train_woman, x_test_woman, y_train_woman, y_test_woman = train_test_split(x_woman, y_woman, train_size=0.63, random_state=123)
print(x_train_woman.shape, y_train_woman.shape) # (5978, 100, 100, 3) (5978,)
print(x_test_woman.shape, y_test_woman.shape)   # (3511, 100, 100, 3) (3511,)

train_datagen = ImageDataGenerator(
    # rescale=1./255,           # 처음부터 0~1 사이의 스케일링한 데이터를 달라
    horizontal_flip=True,       # 수평 뒤집기
    vertical_flip=True,         # 수직 뒤집기
    width_shift_range=0.2,      # 평행 이동, 10%만큼 옆으로 이동
    # height_shift_range=0.1,   # 평행 이동 수직
    rotation_range=15,          # 정해진 각도만큼 이미지 회전
    # zoom_range=1.2,           # 축소 또는 확대
    # shear_range=0.7,          # 좌표 하나를 고정시키고 다른 몇개의 좌표를 이동시키는 변환 (일그러짐)
    fill_mode='nearest',        # 주변의 이미지의 비율대로 채운다
)

augment_size = 8164

randidx = np.random.randint(x_train_woman.shape[0], size=augment_size)
# print(randidx)
# print(np.min(randidx), np.max(randidx))
print(x_train_woman[0].shape)   # (100, 100, 3)

x_augmented = x_train_woman[randidx].copy()
y_augmented = y_train_woman[randidx].copy()
print(x_augmented.shape, y_augmented.shape) # (8164, 100, 100, 3) (8164,)

x_augmented = train_datagen.flow(
    x_augmented, y_augmented,
    batch_size=augment_size,
    shuffle=False,
    save_to_dir='C:\\ai5\\_data\\_save_img\\06_gender\\'
).next()[0]

print(x_augmented.shape)    # (8164, 100, 100, 3)
