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

np.random.seed(337)         # numpy seed 고정
import tensorflow as tf
tf.random.set_seed(337)     # tensorflow seed 고정
import random as rn
rn.seed(337)                # python seed 고정

#1. 데이터
np_path = 'C:\\ai5\\_data\\_save_npy\\keras45\\'
x_train = np.load(np_path + 'keras45_02_horse_x_train.npy')
y_train = np.load(np_path + 'keras45_02_horse_y_train.npy')

print(x_train.shape, y_train.shape)     # (1027, 100, 100, 3) (1027,)

x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, train_size=0.8, random_state=123)
print(x_train.shape, y_train.shape)     # (821, 100, 100, 3) (821,)
print(x_test.shape, y_test.shape)       # (206, 100, 100, 3) (206,)

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

augment_size = 9179

randidx = np.random.randint(x_train.shape[0], size=augment_size)
# print(randidx)
# print(np.min(randidx), np.max(randidx))
print(x_train[0].shape)     # (100, 100, 3)

x_augmented = x_train[randidx].copy()       # .copy() 하면 메모리공간 새로 할당, 영향을 미치지 않는다, 비파괴적
y_augmented = y_train[randidx].copy()       # x, y 순서 주의
print(x_augmented.shape, y_augmented.shape) # (9179, 100, 100, 3) (9179,)

# x_augmented = x_augmented.reshape(
#     x_augmented.shape[0],               # 40000
#     x_augmented.shape[1],               # 28
#     x_augmented.shape[2], 3)            # 28
# print(x_augmented.shape)                # (9179, 100, 100, 3)

x_augmented = train_datagen.flow(
    x_augmented, y_augmented,
    batch_size=augment_size,
    shuffle=False,
).next()[0]
print(x_augmented.shape)                # (9179, 100, 100, 3)

# # [검색] 넘파이 행렬 데이터 합치기
x_train = np.concatenate((x_train, x_augmented))
y_train = np.concatenate((y_train, y_augmented))
print(x_train.shape, y_train.shape)     # (10000, 100, 100, 3) (10000,)

print(np.unique(y_train, return_counts=True))
# (array([0., 1.], dtype=float32), array([4834, 5166], dtype=int64))

x_train = x_train.reshape(10000, 100, 100*3)
x_test = x_test.reshape(206, 100, 100*3)

lr = [0.1, 0.01, 0.005, 0.001, 0.0005, 0.0001]
results = []

for i in range(len(lr)):

    #2. 모델 구성
    model = Sequential()
    model.add(Conv1D(filters=16, kernel_size=3, input_shape=(100, 100*3), activation='relu'))
    model.add(Conv1D(32, 2))
    model.add(Dropout(0.2))
    # model.add(Conv2D(16, (3,3), activation='relu', input_shape=(100, 100, 3)))
    model.add(Flatten())
    model.add(Dense(units=16, activation='relu'))
    model.add(Dense(units=8, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))

    #3. 컴파일, 훈련
    from tensorflow.keras.optimizers import Adam
    model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=lr[i]), metrics=['accuracy'])

    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
    es = EarlyStopping(monitor='val_loss', mode='min', patience=20, verbose=1, restore_best_weights=True,)
    rlr = ReduceLROnPlateau(monitor='val_loss', mode='auto', patience=15, verbose=1, factor=0.7)

    ########## mcp 세이브 파일명 만들기 시작 ##########
    import datetime
    date = datetime.datetime.now()
    date = date.strftime("%m%d_%H%M")

    path = './_save/keras69/'
    filename = '{epoch:04d}_valloss_{val_loss:.4f}.hdf5'
    filepath = "".join([path, 'k69_09_date_', str(i+1), '_', date, '_epo_', filename])

    ########## mcp 세이브 파일명 만들기 끝 ##########

    mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1, save_best_only=True, filepath=filepath)

    start2 = time.time()
    hist = model.fit(x_train, y_train, epochs=500, batch_size=32, validation_split=0.2, callbacks=[es, mcp])
    end2 = time.time()

    #4. 평가, 예측
    loss = model.evaluate(x_test, y_test, verbose=1)
    # print("loss : ", loss[0])
    # print("accuracy : ", round(loss[1], 3))

    y_predict = model.predict(x_test)
    # print(y_predict)            # float 형
    # print(y_predict.shape)      # (10000, 10)

    # y_predict = np.round(y_predict)
    # acc = accuracy_score(y_test, y_predict)
    # print('accuracy_score :', acc)
    # print("time :", round(end2 - start2, 2),'초')

    r2 = r2_score(y_test, y_predict)

    print('{0} > loss : {1} / r2 : {2}'.format(lr[i], loss, r2))

'''
0.0001 > loss : [0.6399123072624207, 0.7572815418243408] / r2 : 0.10555756092071533


'''
