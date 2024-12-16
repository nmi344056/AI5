from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM, GRU
from tensorflow.keras.layers import Conv1D, Conv2D, Flatten, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from sklearn.metrics import r2_score
from sklearn.datasets import fetch_california_housing
import numpy as np
import time

#1. 데이터
datasets = fetch_california_housing()
x = datasets.data
y = datasets.target
print(x.shape)      # (20640, 8)

x = x/255.
x = x.reshape(20640, 4, 2)
print(x.shape)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=123)

scaler = RobustScaler()

# #2. 모델구성
model = Sequential()
model.add(LSTM(64, input_shape=(4,2), return_sequences=True, activation='relu'))
model.add(LSTM(64))
# model.add(Conv2D(64, (3,3), activation='relu', input_shape=(2, 2, 2), strides=1, padding='same'))
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
filepath = "".join([path, 'k59_02_date_', date, '_epo_', filename])

########## mcp 세이브 파일명 만들기 끝 ##########

mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1, save_best_only=True, filepath=filepath)

start = time.time()
hist = model.fit(x_train, y_train, epochs=1000, batch_size=100, validation_split=0.2, callbacks=[es, mcp])
end = time.time()

#4. 평가,예측
print("++++++++++++++++++++")
loss = model.evaluate(x_test, y_test)
print("loss : ", loss)

y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print("r2 score : ", r2)
print("time : ", round(end - start, 2), "초")

'''
128 64 32 32 32 1 / train_size=0.7, random_state=123 / epochs=100, batch_size=100

Epoch 77/100
108/116 [==========================>...] - ETA: 0s - loss: 0.5465Restoring model weights from the end of the best epoch: 27.

Epoch 00077: val_loss did not improve from 0.52523
116/116 [==============================] - 0s 609us/step - loss: 0.5416 - val_loss: 1.3223
Epoch 00077: early stopping
++++++++++++++++++++
194/194 [==============================] - 0s 250us/step - loss: 0.5225
loss :  0.522547721862793
r2 score :  0.6048158187177399

++++++++++++++++++++
loss :  0.7035481333732605
r2 score :  0.4679315063315159

CPU
loss :  1.432392954826355
r2 score :  -0.0832674441425354
time :  13.86 초

GPU
loss :  0.5826069116592407
r2 score :  0.5593952113122644
time :  34.85 초

LSTM
loss :  1.3227450847625732
r2 score :  -0.000344778997875661
time :  104.26 초

'''
