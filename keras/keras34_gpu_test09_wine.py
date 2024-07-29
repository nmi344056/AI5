import tensorflow as tf
print(tf.__version__)       # 2.7.4

gpus = tf.config.experimental.list_physical_devices('GPU')
print(gpus)
# [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]


# keras22_softmax2_wine copy

import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input
from sklearn.model_selection import train_test_split
import time
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import r2_score, accuracy_score

#1. 데이터
datasets = load_wine()
print(datasets)
print(datasets.DESCR)
print(datasets.feature_names)

x = datasets.data
y = datasets.target
print(x.shape, y.shape)     # (178, 13) (178,)

print(y)
print(np.unique(y, return_counts=True))
print(pd.value_counts(y))

y = pd.get_dummies(y)
print(y.shape)      # (178, 3)

x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.9,
                                                    random_state=2321, stratify=y)


from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
# scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
scaler = RobustScaler()


scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

print('x_train :', x_train)
print(np.min(x_train), np.max(x_train))
print(np.min(x_test), np.max(x_test))

#2. 모델구성
# model = Sequential()
# model.add(Dense(8, activation='relu', input_dim=13))
# model.add(Dense(16, activation='relu'))
# model.add(Dense(32, activation='relu'))
# model.add(Dense(64, activation='relu'))
# model.add(Dropout(0.3))
# model.add(Dense(64, activation='relu'))
# model.add(Dropout(0.3))
# model.add(Dense(64, activation='relu'))
# model.add(Dropout(0.3))
# model.add(Dense(64, activation='relu'))
# model.add(Dropout(0.3))
# model.add(Dense(64, activation='relu'))
# model.add(Dropout(0.3))
# model.add(Dense(32, activation='relu'))
# model.add(Dense(16, activation='relu'))
# model.add(Dense(8, activation='relu'))

# model.add(Dense(3, activation='softmax'))


#2-2모델구성(함수형)
input1 = Input(shape=(13, ))
dense1 = Dense(8,activation='relu',name='ys1')(input1)
dense2 = Dense(16, activation='relu',name='ys2')(dense1)
dense3 = Dense(32, activation='relu', name='ys3')(dense2)
dense4 = Dense(64, activation='relu', name='ys4')(dense3)
drop2 = Dropout(0.3)(dense4)
dense5 = Dense(64, activation='relu', name='ys5')(drop2)
drop3 = Dropout(0.3)(dense5)
dense6 = Dense(64, activation='relu', name='ys6')(drop3)
drop4 = Dropout(0.3)(dense6)
dense7 = Dense(64, activation='relu', name='ys7')(drop4)
drop5 = Dropout(0.3)(dense7)
dense10 = Dense(64, activation='relu', name='ys10')(drop5)
drop1 = Dropout(0.3)(dense10)

dense11= Dense(32, activation='relu', name='ys11')(drop1)
dense12 = Dense(16, activation='relu', name='ys12')(dense11)
dense13= Dense(8, activation='relu', name='ys13')(dense12)

output1 = Dense(3, activation='softmax', name='ys15')(dense13)
model=Model(inputs=input1, outputs=output1)
model.summary()








#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
start = time.time()

es = EarlyStopping(
    monitor = 'val_loss',
    mode = 'min',
    patience = 100,
    restore_best_weights = True
)


import datetime
date = datetime.datetime.now()       # 현재시간 저장
print(date)         # 2024-07-26 16:50:37.570567
print(type(date))   # <class 'datetime.datetime'>
date = date.strftime("%m%d_%H%M")
print(date)
print(type(date)) # <class 'str'>


path ='C:/AI5/_save/keras32_mcp/09_wine/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'    # 1000-0.7777.hdf5    { } = dictionary, 키와 밸류  d는 정수, f는 소수
filepath = "".join([path, 'k32_09', date, '_', filename])      # 파일위치와 이름을 에포와 발로스로 나타내준다
# 생성 예 : ""./_save/keras29_mcp/k29_1000-0.7777.hdf5"
##################### MCP 세이브 파일명 만들기 끝 ###########################

mcp = ModelCheckpoint(
    monitor='val_loss',
    mode='auto',
    verbose = 1,
    save_best_only=True,
    filepath = filepath
)
hist = model.fit(x_train, y_train, epochs=1000, batch_size=4,
                 validation_split=0.2, callbacks=[es, mcp])
end = time.time()

#4. 평가, 예측
loss = model.evaluate(x_test, y_test, verbose=1)

y_pred = model.predict(x_test)
print(y_pred)
y_pred = np.round(y_pred)
print(y_pred)

accuracy_score = accuracy_score(y_test, y_pred)
print('acc_score :', accuracy_score)
# print('걸린시간 :', round(end - start, 2), '초')
print('로스 :', loss)
print('acc :', round(loss[1],3))

# acc_score : 0.8333333333333334
# 로스 : [0.22132667899131775, 0.8333333134651184]

# Dropout 적용
# 로스 : [0.7467778921127319, 0.8888888955116272]
# acc : 0.889

# 로스 : [0.0, 1.0]
# acc : 1.0

if(gpus):
    print("쥐피유 돈다!!!")
else:
    print("쥐피유 없다! xxxxx")

print('걸린시간 :', round(end - start, 2), '초')

# 쥐피유 돈다!!!
# 걸린시간 : 52.1 초

# 쥐피유 없다! xxxxx
# 걸린시간 : 5.97 초