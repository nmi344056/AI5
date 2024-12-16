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

#2. 모델 구성
model = Sequential()
model.add(LSTM(16, input_shape=(100, 100*3), return_sequences=True))
model.add(LSTM(32))
model.add(Dropout(0.2))
# model.add(Conv2D(16, (3,3), activation='relu', input_shape=(100, 100, 3)))
# model.add(Flatten())
model.add(Dense(units=16, activation='relu'))
model.add(Dense(units=8, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))

#3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

es = EarlyStopping(monitor='val_loss', mode='min', patience=20, verbose=1, restore_best_weights=True,)

########## mcp 세이브 파일명 만들기 시작 ##########
import datetime
date = datetime.datetime.now()
print(date)
print(type(date))

date = date.strftime("%m%d.%H%M")
print(date)
print(type(date))

path = './_save/keras49/'
filename = '{epoch:04d}_valloss_{val_loss:.4f}.hdf5'
filepath = "".join([path, 'k49_07_date_', date, '_epo_', filename])

########## mcp 세이브 파일명 만들기 끝 ##########

mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1, save_best_only=True, filepath=filepath)

start2 = time.time()
hist = model.fit(x_train, y_train, epochs=100, batch_size=100, validation_split=0.2, callbacks=[es, mcp])
end2 = time.time()

#4. 평가, 예측
loss = model.evaluate(x_test, y_test, verbose=1)
print("loss : ", loss[0])
print("accuracy : ", round(loss[1], 3))

y_predict = model.predict(x_test)
print(y_predict)            # float 형
print(y_predict.shape)      # (10000, 10)

y_predict = np.round(y_predict)
acc = accuracy_score(y_test, y_predict)
print('accuracy_score :', acc)
# print("time :", round(end2 - start2, 2),'초')

'''
[실습] accuracy 1.0 이상
loss :  0.07973863184452057 / accuracy :  0.976 > 

augment
loss :  0.033560577780008316
accuracy :  0.99

loss :  0.002064676024019718
accuracy :  1.0

LSTM
loss :  0.18117965757846832
accuracy :  0.971

'''
