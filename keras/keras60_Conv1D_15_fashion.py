# 37_2 copy

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM, GRU
from tensorflow.keras.layers import Conv1D, Conv2D, Flatten, Dropout, MaxPooling2D
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

'''
x_augmented = train_datagen.flow(
    x_augmented, y_augmented,
    batch_size=augment_size,
    shuffle=False,
).next()

# print(x_augmented.shape)
# ValueError: ('Input data in `NumpyArrayIterator` should have rank 4. You passed an array with shape', (40000, 28, 28))
print(x_augmented[0].shape)     # (40000, 28, 28, 1)
'''

x_augmented = train_datagen.flow(
    x_augmented, y_augmented,
    batch_size=augment_size,
    shuffle=False,
).next()[0]

# print(x_augmented.shape)
# ValueError: ('Input data in `NumpyArrayIterator` should have rank 4. You passed an array with shape', (40000, 28, 28))
print(x_augmented.shape)     # (40000, 28, 28, 1)

x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(10000, 28, 28, 1)
print(x_train.shape, x_test.shape)      # (60000, 28, 28, 1) (10000, 28, 28, 1)

# [검색] 넘파이 행렬 데이터 합치기
x_train = np.concatenate((x_train, x_augmented))
y_train = np.concatenate((y_train, y_augmented))
print(x_train.shape, y_train.shape)        # (100000, 28, 28, 1) (100000,)

# [실습] 만들기

y_train = pd.get_dummies(y_train)
y_test = pd.get_dummies(y_test)
print(y_train.shape, y_test.shape)      # (100000, 10) (10000, 10)

#2. 모델 구성
model = Sequential()
model.add(Conv1D(filters=64, kernel_size=3, input_shape=(28, 28), activation='relu'))
model.add(Conv1D(64, 2))
model.add(Dropout(0.2))
# model.add(Conv2D(64, (3,3), activation='relu', input_shape=(28, 28, 1), strides=1, padding='same'))
model.add(Flatten())
model.add(Dense(units=32, activation='relu'))
model.add(Dense(units=32, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(units=16, input_shape=(32,), activation='relu'))
model.add(Dense(10, activation='softmax'))

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

es = EarlyStopping(monitor='val_loss', mode='min', patience=10, verbose=1, restore_best_weights=True,)

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
filepath = "".join([path, 'k59_01_date_', date, '_epo_', filename])

########## mcp 세이브 파일명 만들기 끝 ##########

mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1, save_best_only=True, filepath=filepath)

start = time.time()
hist = model.fit(x_train, y_train, epochs=100, batch_size=16, validation_split=0.2, callbacks=[es, mcp])
end = time.time()

#4. 평가, 예측
loss = model.evaluate(x_test, y_test, verbose=1)
print("loss : ", loss)

y_predict = model.predict(x_test)
y_predict = np.argmax(y_predict, axis=1).reshape(-1,1)

#print(y_test)로 데이터 구조? 확인 (pandas > numpy)
y_test = y_test.to_numpy()
y_test = np.argmax(y_test, axis=1).reshape(-1,1)

acc = accuracy_score(y_test, y_predict)
print('accuracy_score :', acc)
# print("time :", round(end-start,2),'초')

print("++++++++++++++++++++")
print("loss : ", loss[0], "/ accuracy : ", acc)

'''
[실습] accuracy 0.98 이상
64-3,3  64-3,3-0.2  32-2.2  32-0.2  16 10 + relu / patience=50 / epochs=1000, batch_size=16 > accuracy_score : 
loss :  [0.2626487910747528, 0.9088000059127808]


stride_padding


MaxPooling
loss :  0.2265615016222 / accuracy :  0.9199
loss :  0.23189404606819153 / accuracy :  0.9193

augment
loss :  0.23711518943309784 / accuracy :  0.9151

LSTM
loss :  0.30284762382507324 / accuracy :  0.8933

Conv1D
loss :  0.3194533586502075 / accuracy :  0.8886

'''
