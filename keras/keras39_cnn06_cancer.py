from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from sklearn.metrics import r2_score, accuracy_score
from sklearn.datasets import load_breast_cancer
import numpy as np
import pandas as pd
import time

#1. 데이터
datasets = load_breast_cancer()

x = datasets.data
y = datasets.target
print(x.shape, y.shape)                     # (569, 30) (569,)

x = x/255.
x = x.reshape(569, 10, 3, 1)
print(x.shape)


x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=555)

scaler = MaxAbsScaler()

#2. 모델구성
model = Sequential()
model.add(Conv2D(64, (3,3), activation='relu', input_shape=(10, 3, 1), strides=1, padding='same'))
model.add(Dropout(0.3))
model.add(Conv2D(filters=64, kernel_size=(3,3), activation='relu', strides=1, padding='same'))
model.add(Dropout(0.2))
model.add(Conv2D(32, (2,2), activation='relu', strides=1, padding='same'))
model.add(Flatten())

model.add(Dense(units=32, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(units=16, input_shape=(32,), activation='relu'))
model.add(Dense(10, activation='softmax'))
model.add(Dense(1, activation='sigmoid'))

#3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

es = EarlyStopping(monitor='val_loss', mode='min', patience=50, restore_best_weights=True)

########## mcp 세이브 파일명 만들기 시작 ##########
import datetime
date = datetime.datetime.now()
print(date)
print(type(date))

date = date.strftime("%m%d.%H%M")
print(date)
print(type(date))

path = './_save/keras39/'
filename = '{epoch:04d}_valloss_{val_loss:.4f}.hdf5'
filepath = "".join([path, 'k39_06_date_', date, '_epo_', filename])

########## mcp 세이브 파일명 만들기 끝 ##########

mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1, save_best_only=True, filepath=filepath)

start = time.time()
hist = model.fit(x_train, y_train, epochs=100, batch_size=8, validation_split=0.2, callbacks=[es, mcp])
end = time.time()

#4. 평가, 예측
loss = model.evaluate(x_test, y_test, verbose=1)
print("loss : ", loss[0])
print("accuracy : ", round(loss[1], 3))

y_predict = model.predict(x_test)
print(y_predict[:20])               # y' 결과
y_predict = np.round(y_predict)
print(y_predict[:20])               # y' 반올림 결과

accuracy_score = accuracy_score(y_test, y_predict)
print("acc score : ", accuracy_score)
print("time : ", round(end - start, 2), "초")

print("===============")
print("loss : ", loss[0])
print("accuracy : ", round(loss[1], 3))

'''
63 32 32 32 32 1 / train_size=0.8, random_state=555 / epochs=100, batch_size=8 / verbose=1
mse / loss : 

loss :  0.10229721665382385
accuracy :  0.974
'''
