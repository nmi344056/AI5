from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from sklearn.metrics import r2_score
from sklearn.datasets import load_diabetes
import numpy as np
import time

#1. 데이터
datasets = load_diabetes()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=999)

scaler = MaxAbsScaler()

x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델구성
input1 = Input(shape=(10,))
dense1 = Dense(100)(input1)
drop1 = Dropout(0.3)(dense1)
dense2 = Dense(75)(drop1)
drop2 = Dropout(0.3)(dense2)
dense3 = Dense(50)(drop2)
drop3 = Dropout(0.2)(dense3)
dense4 = Dense(30)(drop3)
output1 = Dense(1)(dense4)
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
filepath = "".join([path, 'k32_03_date_', date, '_epo_', filename])

########## mcp 세이브 파일명 만들기 끝 ##########

mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1, save_best_only=True, filepath=filepath)

start = time.time()
hist = model.fit(x_train, y_train, epochs=300, batch_size=10, validation_split=0.2, callbacks=[es, mcp])
end = time.time()

#4.평가, 예측
print("++++++++++++++++++++")
loss = model.evaluate(x_test, y_test)
print("loss : ", loss)

y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print("r2 sorce : ", r2)
print("time : ", round(end - start, 2), "초")

'''
100 75 50 30 1 / train_size=0.8, random_state=999 / epochs=100, batch_size=10

Epoch 95/300
 1/29 [>.............................] - ETA: 0s - loss: 3317.6062Restoring model weights from the end of the best epoch: 45.

Epoch 00095: val_loss did not improve from 2776.82617
29/29 [==============================] - 0s 821us/step - loss: 3326.5530 - val_loss: 2841.9988
Epoch 00095: early stopping
++++++++++++++++++++
3/3 [==============================] - 0s 500us/step - loss: 2197.2817
loss :  2197.28173828125
r2 sorce :  0.596755557322737

loss :  2156.3740234375
r2 sorce :  0.6042629302834428

CPU
loss :  2181.60595703125
r2 sorce :  0.5996323888389401
time :  4.48 초

GPU
loss :  2328.738525390625
r2 sorce :  0.5726306939555827
time :  9.09 초
'''
