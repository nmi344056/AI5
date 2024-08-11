# https://dacon.io/competitions/open/235576/data

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM, GRU
from tensorflow.keras.layers import Conv1D, Conv2D, Flatten, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from sklearn.metrics import r2_score
import numpy as np
import pandas as pd
import time

#1. 데이터
path = "./_data/dacon/따릉이/"

train_csv = pd.read_csv(path + "train.csv", index_col=0)
print(train_csv)            # [1459 rows x 11 columns] / [1459 rows x 10 columns]

test_csv = pd.read_csv(path + "test.csv", index_col=0)
print(test_csv)             # [715 rows x 9 columns]

submission_csv = pd.read_csv(path + "submission.csv", index_col=0)
print(submission_csv)       # [715 rows x 1 columns]

########## 결측치 처리 1. 삭제 ##########
# print(train_csv.isnull().sum())
print(train_csv.isna().sum())

train_csv = train_csv.dropna()
print(train_csv.isna().sum())
print(train_csv)            # [1328 rows x 10 columns]
print(train_csv.info())

print(test_csv.info())

test_csv = test_csv.fillna(test_csv.mean())
print(test_csv.info())

x = train_csv.drop(['count'], axis=1)
print(x)                    # [1328 rows x 9 columns]
y = train_csv['count']
print(y)
print(y.shape)              # (1328,)

x = x.to_numpy()
print(x)

x = x/255.
x = x.reshape(1328, 3, 3)
print(x.shape)      # (1328, 3, 3)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.9, random_state=4343)

scaler = MinMaxScaler()

# #2. 모델구성
model = Sequential()
model.add(LSTM(64, input_shape=(3,3), return_sequences=True, activation='relu'))
model.add(LSTM(64, activation='relu'))
# model.add(Conv2D(64, (3,3), activation='relu', input_shape=(3, 3, 1), strides=1, padding='same'))
# model.add(Flatten())
model.add(Dense(units=32, activation='relu'))
model.add(Dense(units=32, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(units=16, input_shape=(32,), activation='relu'))
model.add(Dense(10, activation='softmax'))
model.add(Dense(1, activation='linear'))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

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
filepath = "".join([path, 'k59_04_data_', date, '_epo_', filename])

########## mcp 세이브 파일명 만들기 끝 ##########

mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1, save_best_only=True, filepath=filepath)

start = time.time()
hist = model.fit(x_train, y_train, epochs=1000, batch_size=24, validation_split=0.2, callbacks=[es, mcp])
end = time.time()

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print("loss : ", loss)

y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print("r2 score : ", r2)
print("time : ", round(end - start, 2), "초")

print(test_csv)
test_csv = test_csv.to_numpy()
print(test_csv.shape)       # (715, 9)
test_csv = test_csv.reshape(715, 3, 3, 1)
print(test_csv.shape)       # (715, 3, 3, 1)

y_submit = model.predict(test_csv)
print(y_submit)
print(y_submit.shape)       # (715, 1)

########## submission.csv 만들기 (count컬럼에 값만 넣으면 된다) ##########
submission_csv['count'] = y_submit
print(submission_csv)           # [715 rows x 1 columns]
print(submission_csv.shape)     # (715, 1)

submission_csv.to_csv(path + "submission_0801_12.csv")

'''
126 64 32 32 32 1 / train_size=0.9, random_state=4343 / epochs=500, batch_size=24

loss :  1641.72021484375
r2 score :  0.7043070084804695

LSTM
loss :  6364.99267578125
r2 score :  -0.14640961762537263
(과적합, 수정해서 다시 돌리기)

'''
