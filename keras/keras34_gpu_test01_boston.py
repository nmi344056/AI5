# [복사] keras29_5.py

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score
from sklearn.datasets import load_boston
import numpy as np
import time

#1. 데이터
dataset = load_boston()

x = dataset.data
y = dataset.target

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=555)

from sklearn.preprocessing import MinMaxScaler, StandardScaler
scaler = MinMaxScaler()

x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델구성
input1 = Input(shape=(13,))
dense1 = Dense(64)(input1)
drop1 = Dropout(0.3)(dense1)
dense2 = Dense(64)(drop1)
drop2 = Dropout(0.3)(dense2)
dense3 = Dense(32)(drop2)
drop3 = Dropout(0.2)(dense3)
dense4 = Dense(32)(drop3)
drop4 = Dropout(0.1)(dense4)
dense5 = Dense(26)(drop4)
output1 = Dense(1)(dense5)
model = Model(inputs=input1, outputs=output1)

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='val_loss', mode='min', patience=10, verbose=1, restore_best_weights=True,)

########## mcp 세이브 파일명 만들기 시작 ##########
import datetime
date = datetime.datetime.now()
print(date)             # 2024-07-26 16:54:35.479745
print(type(date))       # <class 'datetime.datetime'>

date = date.strftime("%m%d_%H%M")
print(date)             # 0726_1654
print(type(date))       # <class 'str'>

path = './_save/keras32/'
filename = '{epoch:04d}_valloss_{val_loss:.4f}.hdf5'
filepath = "".join([path, 'k32_01_date_', date, '_epo_', filename])

########## mcp 세이브 파일명 만들기 끝 ##########

mcp = ModelCheckpoint(monitor='val_loss', mode='auto',
                      verbose=1,
                      save_best_only=True,
                      filepath=filepath
                      )

start = time.time()
hist = model.fit(x_train, y_train, epochs=1000, batch_size=10, validation_split=0.2, callbacks=[es, mcp])
end = time.time()

# model.save('./_save/keras29_mcp/keras29_3_save_model.hdf5')

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print("loss : ", loss)

y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print("r2 score : ", r2)
print("time : ", round(end - start, 2), "초")

'''
loss :  24.317855834960938
r2 score :  0.7176152349757086
++++++++++++++++++++
dropout
loss :  15.876220703125
r2 score :  0.8156414819858147

CPU
loss :  22.616262435913086
r2 score :  0.737374485062637
time :  1.91 초

GPU
loss :  24.234468460083008
r2 score :  0.7185835076684911
time :  4.44 초
'''
