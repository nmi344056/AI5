from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM, GRU
from tensorflow.keras.layers import Conv1D, Conv2D, Flatten, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from sklearn.metrics import r2_score, accuracy_score
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.datasets import load_digits
import numpy as np
import pandas as pd
import time

#1. 데이터
x, y = load_digits(return_X_y=True)     #로 분할 가능
print(x)                    # [[...]...[...]]
print(y)                    # [0 1 2 ... 8 9 8]
print(x.shape, y.shape)     # (1797, 64) (1797,)

print(np.unique(y, return_counts=True))
# (array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]), array([178, 182, 177, 183, 181, 182, 181, 179, 174, 180], dtype=int64))

x = x/255.
x = x.reshape(1797, 8, 8)
print(x.shape)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.9, random_state=6666)

scaler = MaxAbsScaler()

#2. 모델구성
model = Sequential()
model.add(Conv1D(filters=64, kernel_size=3, input_shape=(8,8), activation='relu'))
model.add(Conv1D(64, 2))
# model.add(Conv2D(64, (3,3), activation='relu', input_shape=(8, 8, 1), strides=1, padding='same'))
model.add(Flatten())
model.add(Dense(units=32, activation='relu'))
model.add(Dense(units=32, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(units=16, input_shape=(32,), activation='relu'))
model.add(Dense(1, activation='linear'))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

es = EarlyStopping(monitor='val_loss', mode='min', patience=50, verbose=1, restore_best_weights=True,)

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
filepath = "".join([path, 'k59_11_digits_date_', date, '_epo_', filename])

########## mcp 세이브 파일명 만들기 끝 ##########

mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1, save_best_only=True, filepath=filepath)

start = time.time()
hist = model.fit(x_train, y_train, epochs=1000, batch_size=100, validation_split=0.2, callbacks=[es, mcp])
end = time.time()

#4. 평가, 예측
loss = model.evaluate(x_test, y_test, verbose=1)
print("loss : ", loss[0])
print("accuracy : ", round(loss[1], 3))

y_predict = model.predict(x_test)
# print(y_predict[:20])
y_predict = np.round(y_predict)
# print(y_predict[:20])

accuracy_score = accuracy_score(y_test, y_predict)
print("acc score : ", accuracy_score)
print("time : ", round(end - start, 2), "초")

'''
[실습] accuracy :  1.0 이상
128 256 256 256 128 10 / train_size=0.9, random_state=6666 / epochs=100, batch_size=100

loss :  0.11226184666156769
accuracy :  0.978

CPU
loss :  0.19280655682086945
accuracy :  0.978
time :  4.25 초

GPU
loss :  0.1523342728614807
accuracy :  0.972
time :  5.93 초

LSTM
loss :  1.1853468418121338
accuracy :  0.178
acc score :  0.49444444444444446
time :  143.26 초

Conv1D
loss :  0.5402076244354248
accuracy :  0.172
acc score :  0.5944444444444444
time :  20.1 초

'''
