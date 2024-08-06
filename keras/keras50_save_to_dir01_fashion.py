# 49_1 copy

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, Input, MaxPooling2D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from sklearn.metrics import r2_score, accuracy_score

#1. 데이터
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

x_train = x_train/255.
x_test = x_test/255.

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

augment_size = 40000    # 6만개를 10만개로 늘리기 (4만개 추가)

print(x_train.shape[0]) # 60000
print(x_train.shape[1]) # 28
print(x_train.shape[2]) # 28
# print(x_train.shape[3]) # IndexError: tuple index out of range

randidx = np.random.randint(x_train.shape[0], size=augment_size)
print(randidx)          # [44199 54559 36838 ... 49240 16552 36947] 랜덤한 값
print(np.min(randidx), np.max(randidx))     # 0 59998 (60000 이상만 안나오면 정상)
print(x_train[0].shape)     # (28, 28)      이미지 한 장

x_augmented = x_train[randidx].copy()       # .copy() 하면 메모리공간 새로 할당, 영향을 미치지 않는다, 비파괴적
y_augmented = y_train[randidx].copy()       # x, y 순서 주의
print(x_augmented.shape, y_augmented.shape) # (40000, 28, 28) (40000,)

# x_augmented = x_augmented.reshape(40000, 28, 28, 1)     # (40000, 28, 28, 1)으로작성하면 바꿔야한다
x_augmented = x_augmented.reshape(
    x_augmented.shape[0],   # 40000
    x_augmented.shape[1],   # 28
    x_augmented.shape[2], 1)   # 28
print(x_augmented.shape)    # (40000, 28, 28, 1)

x_augmented = train_datagen.flow(
    x_augmented, y_augmented,
    batch_size=augment_size,
    shuffle=False,
    save_to_dir='C:\\ai5\\_data\\_save_img\\01_fashion\\'
).next()[0]

# print(x_augmented.shape)
# ValueError: ('Input data in `NumpyArrayIterator` should have rank 4. You passed an array with shape', (40000, 28, 28))
print(x_augmented.shape)     # (40000, 28, 28, 1)
