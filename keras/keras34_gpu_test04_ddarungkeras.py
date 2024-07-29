# https://dacon.io/competitions/open/235576/data

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Dropout
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

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.9, random_state=4343)

scaler = MinMaxScaler()

x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
test_csv = scaler.transform(test_csv)       # 추가

print(x_train)
print(np.min(x_train), np.max(x_train))     # 0.0 1.0
print(np.min(x_test), np.max(x_test))       # 0.0 1.0

#2. 모델구성
input1 = Input(shape=(9,))
dense1 = Dense(126)(input1)
drop1 = Dropout(0.3)(dense1)
dense2 = Dense(64)(drop1)
drop2 = Dropout(0.3)(dense2)
dense3 = Dense(32)(drop2)
drop3 = Dropout(0.2)(dense3)
dense4 = Dense(32)(drop3)
drop4 = Dropout(0.1)(dense4)
dense5 = Dense(32)(drop4)
output1 = Dense(1)(dense5)
model = Model(inputs=input1, outputs=output1)

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

path = './_save/keras32/'
filename = '{epoch:04d}_valloss_{val_loss:.4f}.hdf5'
filepath = "".join([path, 'k32_04_data_', date, '_epo_', filename])

########## mcp 세이브 파일명 만들기 끝 ##########

mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1, save_best_only=True, filepath=filepath)

start = time.time()
hist = model.fit(x_train, y_train, epochs=500, batch_size=24, validation_split=0.2, callbacks=[es, mcp])
end = time.time()

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print("loss : ", loss)

y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print("r2 score : ", r2)
print("time : ", round(end - start, 2), "초")

# y_submit = model.predict(test_csv)
# print(y_submit)
# print(y_submit.shape)           # (715, 1)

# ########## submission.csv 만들기 (count컬럼에 값만 넣으면 된다) ##########
# submission_csv['count'] = y_submit
# print(submission_csv)           # [715 rows x 1 columns]
# print(submission_csv.shape)     # (715, 1)

# submission_csv.to_csv(path + "submission_0725_18.csv")

'''
126 64 32 32 32 1 / train_size=0.9, random_state=4343 / epochs=500, batch_size=24

loss :  1641.72021484375
r2 score :  0.7043070084804695

CPU
loss :  1633.5167236328125
r2 score :  0.7057845324213317
time :  3.24 초

GPU
loss :  1642.0316162109375
r2 score :  0.7042509177951755
time :  13.82 초
'''
