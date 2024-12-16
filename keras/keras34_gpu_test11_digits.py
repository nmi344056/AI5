from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Dropout
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
print(x)                    # [[]...[]]
print(y)                    # [0 1 2 ... 8 9 8]
print(x.shape, y.shape)     # (1797, 64) (1797,)

print(pd.value_counts(y, sort=False))
# 0    178
# 1    182
# 2    177
# 3    183
# 4    181
# 5    182
# 6    181
# 7    179
# 8    174
# 9    180

print(np.unique(y, return_counts=True))
# (array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]), array([178, 182, 177, 183, 181, 182, 181, 179, 174, 180], dtype=int64))

y_ohe1 = to_categorical(y)
print(y_ohe1)
print(y_ohe1.shape)         # (1797, 10)

x_train, x_test, y_train, y_test = train_test_split(x, y_ohe1, train_size=0.9, random_state=6666)

scaler = MaxAbsScaler()

scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

print(x_train)
print(np.min(x_train), np.max(x_train))     # 0.0 1.0
print(np.min(x_test), np.max(x_test))       # 0.0 1.0

print(x_train.shape, x_test.shape)      # (1617, 64) (180, 64)
print(y_train.shape, y_test.shape)      # (1617, 10) (180, 10)

#2. 모델구성
input1 = Input(shape=(64,))
dense1 = Dense(128, activation='relu')(input1)
drop1 = Dropout(0.3)(dense1)
dense2 = Dense(256, activation='relu')(drop1)
drop2 = Dropout(0.3)(dense2)
dense3 = Dense(256, activation='relu')(drop2)
drop3 = Dropout(0.2)(dense3)
dense4 = Dense(256, activation='relu')(drop3)
drop4 = Dropout(0.1)(dense4)
dense5 = Dense(128, activation='relu')(drop4)
output1 = Dense(10, activation='softmax')(dense5)
model = Model(inputs=input1, outputs=output1)

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

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
filepath = "".join([path, 'k32_11_digits_date_', date, '_epo_', filename])

########## mcp 세이브 파일명 만들기 끝 ##########

mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1, save_best_only=True, filepath=filepath)

start = time.time()
hist = model.fit(x_train, y_train, epochs=100, batch_size=100, validation_split=0.2, callbacks=[es, mcp])
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
'''
