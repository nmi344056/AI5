from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Dropout
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

print(np.unique(y))                         # [0 1] 꼭 확인
print(np.unique(y, return_counts=True))     # (array([0, 1]), array([212, 357], dtype=int64))

print(type(x))                              # <class 'numpy.ndarray'>
print(pd.DataFrame(y).value_counts())
# 1    357
# 0    212

# print(y.value_counts())                   # AttributeError: 'numpy.ndarray' object has no attribute 'value_counts'

print(pd.Series(y))
print("++++++++++++++++++++")
print(pd.Series(y).value_counts())
print(pd.value_counts(y))

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=555)

scaler = MaxAbsScaler()

x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

print(x_train)
print(np.min(x_train), np.max(x_train))     # 0.0 1.0
print(np.min(x_test), np.max(x_test))       # -0.0008218036468019101 1.5452539790933444

print(x_train.shape, y_train.shape)         # (455, 30) (455,)
print(x_test.shape, y_test.shape)           # (114, 30) (114,)

#2. 모델구성
input1 = Input(shape=(30,))
dense1 = Dense(63)(input1)
drop1 = Dropout(0.3)(dense1)
dense2 = Dense(32, activation='relu')(drop1)
drop2 = Dropout(0.3)(dense2)
dense3 = Dense(32, activation='relu')(drop2)
drop3 = Dropout(0.2)(dense3)
dense4 = Dense(32, activation='relu')(drop3)
drop4 = Dropout(0.2)(dense4)
dense5 = Dense(32, activation='relu')(drop4)
output1 = Dense(1, activation='sigmoid')(dense5)
model = Model(inputs=input1, outputs=output1)

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

path = './_save/keras32/'
filename = '{epoch:04d}_valloss_{val_loss:.4f}.hdf5'
filepath = "".join([path, 'k32_06_date_', date, '_epo_', filename])

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

CPU
loss :  0.18007823824882507
acc score :  0.9649122807017544
time :  3.51 초

GPU
loss :  0.1707141101360321
acc score :  0.9385964912280702
time :  20.01 초
'''
