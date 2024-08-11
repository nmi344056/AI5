from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM, GRU
from tensorflow.keras.layers import Conv1D, Conv2D, Flatten, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from sklearn.metrics import r2_score, accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.datasets import fetch_covtype
import numpy as np
import pandas as pd
import time

#1. 데이터
datasets = fetch_covtype()
x = datasets.data
y = datasets.target
print(x.shape, y.shape)     # (581012, 54) (581012,)

print(pd.value_counts(y))    

print(y)
print(np.unique(y, return_counts=True))

x = x/255.
x = x.reshape(581012, 9, 6)
print(x.shape)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.9, random_state=6666,
                                                    stratify=y)

scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
# scaler = RobustScaler()

#2. 모델구성
model = Sequential()
model.add(LSTM(64, input_shape=(9,6), return_sequences=True, activation='relu'))
model.add(LSTM(64, activation='relu'))
# model.add(Conv2D(64, (3,3), activation='relu', input_shape=(9, 6, 1), strides=1, padding='same'))
# model.add(Flatten())
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
filepath = "".join([path, 'k59_10_fetch_covtype_date_', date, '_epo_', filename])

########## mcp 세이브 파일명 만들기 끝 ##########

mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1, save_best_only=True, filepath=filepath)

start = time.time()
hist = model.fit(x_train, y_train, epochs=100, batch_size=300, validation_split=0.2, callbacks=[es, mcp])
end = time.time()

#4. 평가, 예측
loss = model.evaluate(x_test, y_test, verbose=1)
print("loss : ", loss[0])
print("accuracy : ", round(loss[1], 3))

y_predict = model.predict(x_test)
print(y_predict[:20])
y_predict = np.round(y_predict)
print(y_predict[:20])

accuracy_score = accuracy_score(y_test, y_predict)
print("acc score : ", accuracy_score)
print("time : ", round(end - start, 2), "초")

print(x)
'''
128 256 256 256 126 8 / train_size=0.9, random_state=6666 / epochs=100, batch_size=300, validation_split=0.2

loss :  0.28132835030555725
accuracy :  0.888

LSTM
loss :  0.31243953108787537
accuracy :  0.365
acc score :  0.8440845409796565
time :  4204.77 초

'''
