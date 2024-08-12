# 37_1 copy

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM, GRU
from tensorflow.keras.layers import Conv1D, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from sklearn.metrics import r2_score, accuracy_score

#1. 데이터
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

print(x_train.shape, y_train.shape) # (50000, 32, 32, 3) (50000, 1)
print(x_test.shape, y_test.shape)   # (10000, 32, 32, 3) (10000, 1)

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

augment_size = 50000    # 5만개를 10만개로 늘리기 (5만개 추가)

randidx = np.random.randint(x_train.shape[0], size=augment_size)
print(randidx)          # [16378 21212 31335 ... 28117 49292 20443]
print(np.min(randidx), np.max(randidx))     # 0 49998
print(x_train[0].shape)     # (28, 28)      이미지 한 장

x_augmented = x_train[randidx].copy()       # .copy() 하면 메모리공간 새로 할당, 영향을 미치지 않는다, 비파괴적
y_augmented = y_train[randidx].copy()       # x, y 순서 주의
print(x_augmented.shape, y_augmented.shape) # (50000, 32, 32, 3) (50000, 1)

# # x_augmented = x_augmented.reshape(40000, 28, 28, 1)     # (40000, 28, 28, 1)으로작성하면 바꿔야한다
# x_augmented = x_augmented.reshape(
#     x_augmented.shape[0],   # 40000
#     x_augmented.shape[1],   # 28
#     x_augmented.shape[2], 1)   # 28

# print(x_augmented.shape)    # (40000, 28, 28, 1)

x_augmented = train_datagen.flow(
    x_augmented, y_augmented,
    batch_size=augment_size,
    shuffle=False,
).next()[0]

print(x_augmented.shape)     # (50000, 32, 32, 3)

# x_train = x_train.reshape(60000, 28, 28, 1)
# x_test = x_test.reshape(10000, 28, 28, 1)
# print(x_train.shape, x_test.shape)      # (60000, 28, 28, 1) (10000, 28, 28, 1)

# # [검색] 넘파이 행렬 데이터 합치기
x_train = np.concatenate((x_train, x_augmented))
y_train = np.concatenate((y_train, y_augmented))
print(x_train.shape, y_train.shape)        # (100000, 32, 32, 3) (100000, 1)

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
print(y_train.shape, y_test.shape)      # (100000, 10) (10000, 10)

x_train = x_train.reshape(100000, 32, 32*3)
x_test = x_test.reshape(10000, 32, 32*3)

#2. 모델 구성
model = Sequential()
model.add(Conv1D(filters=64, kernel_size=3, input_shape=(32, 32*3), activation='relu'))
model.add(Conv1D(64, 2))
model.add(Dropout(0.2))
# model.add(Conv2D(64, (3,3), activation='relu', input_shape=(32, 32, 3), strides=1, padding='same'))
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(16, activation='relu'))
model.add(Dense(10, activation='softmax'))

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

es = EarlyStopping(monitor='val_loss', mode='min', patience=20, verbose=1, restore_best_weights=True,)

########## mcp 세이브 파일명 만들기 시작 ##########
import datetime
date = datetime.datetime.now()
print(date)
print(type(date))

date = date.strftime("%m%d.%H%M")
print(date)
print(type(date))

path = './_save/keras59/'
filename = '{epoch:04d}_valloss_{val_loss:.4f}.hdf5'
filepath = "".join([path, 'k59_03_date_', date, '_epo_', filename])

########## mcp 세이브 파일명 만들기 끝 ##########

mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1, save_best_only=True, filepath=filepath)

start = time.time()
hist = model.fit(x_train, y_train, epochs=100, batch_size=256, validation_split=0.2, callbacks=[es, mcp])
end = time.time()

#4. 평가, 예측
loss = model.evaluate(x_test, y_test, verbose=1)
print("loss : ", loss[0])
print("accuracy : ", round(loss[1], 3))

y_predict = model.predict(x_test)
# print(y_predict)            # float 형
# print(y_predict.shape)      # (10000, 10)

y_predict = np.argmax(y_predict, axis=1).reshape(-1,1)
print(y_predict)            #  int 형
print(y_predict.shape)      # (10000, 1)

y_test = np.argmax(y_test, axis=1).reshape(-1,1)
print(y_test)
print(y_test.shape)

acc = accuracy_score(y_test, y_predict)
print('accuracy_score :', acc)
# print("time :", round(end-start,2),'초')

print("++++++++++++++++++++")
print("loss : ", loss[0], "/ accuracy : ", round(loss[1], 3))

'''
64-3,3-0.3  64-3,3-0.2  32-2,2  32-0.1  16 10 + relu / patience=10 / epochs=1000, batch_size=16 > accuracy_score : 0.5991
64-3,3-0.3  64-3,3-0.2  32-2,2  32-0.1  16 10 + relu / patience=10 / epochs=1000, batch_size=32 > accuracy_score : 0.6251
64-3,3-0.3  64-3,3-0.2  32-2,2  32-0.1  16 10 + relu / patience=20 / epochs=1000, batch_size=128 > accuracy_score : 0.681
64-3,3-0.3  64-3,3-0.2  32-2,2  32-0.1  16 10 + relu / patience=20 / epochs=1000, batch_size=256 > accuracy_score : 0.6495

loss :  1.0595110654830933 / accuracy :  0.635

stride_padding
loss :  1.0732710361480713 / accuracy :  0.617

augment
loss :  0.7034227252006531 / accuracy :  0.763

LSTM
loss :  1.174163818359375 / accuracy :  0.582

Conv1D
loss :  1.1895502805709839 / accuracy :  0.575

'''
