from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
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
model = Sequential()
model.add(Dense(100, input_shape=(10,)))
model.add(Dense(75))
model.add(Dense(50))
model.add(Dense(30))
model.add(Dense(1))

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

path = './_save/keras30_mcp/'
filename = '{epoch:04d}_valloss_{val_loss:.4f}.hdf5'
filepath = "".join([path, 'k30_03_diabetes_date_', date, '_epo_', filename])
# k30_01_boston_date_0726.2005_epo_0013_valloss_18.3803.hdf5

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
'''
