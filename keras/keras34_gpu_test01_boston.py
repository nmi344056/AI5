# 01 부터 13까지 쭉 카피해서

# gpu일때와 cpu일때의 시간을 체크해서 비교할것

import tensorflow as tf
print(tf.__version__)       # 2.7.4

gpus = tf.config.experimental.list_physical_devices('GPU')
print(gpus)
# [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]

if(gpus):
    print("쥐피유 돈다!!!")
else:
    print("쥐피유 없다! xxxxx")


# 과적합이 문제다

# keras29_MCP5 copy

import numpy as np
import sklearn as sk
print(sk.__version__)   # 0.24.2
from sklearn.datasets import load_boston
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import time

#1. 데이터
dataset = load_boston()
print(dataset)
print(dataset.DESCR)   # DESCR = pandas 의 describe
print(dataset.feature_names)
# ['CRIM' 'ZN' 'INDUS' 'CHAS' 'NOX' 'RM' 'AGE' 'DIS' 'RAD' 'TAX' 'PTRATIO'
#  'B' 'LSTAT']

x = dataset.data
y = dataset.target

print(x)
print(x.shape)      #(506, 13)    --> input_dim=13
print(y)
print(y.shape)      #(506,)  벡터

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.9,
                                                    random_state=6666)

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
# scaler = MinMaxScaler()
# scaler = StandardScaler()
scaler = MaxAbsScaler()
# scaler = RobustScaler()

# scaler.fit(x_train)
# x_train = scaler.transform(x_train)
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

print('x_train :', x_train)
print(np.min(x_train), np.max(x_train))
print(np.min(x_test), np.max(x_test))


# print('x_test :', x_test)
# print('y_train :', y_train)
# print('y_test :', y_test)


# #2. 모델구성
# model = Sequential()
# # model.add(Dense(10, input_dim=13))
# model.add(Dense(32, input_shape=(13,)))   # input_shape 는 벡터형태로  # 이미지 input_shape=(8,8,1)
# model.add(Dropout(0.3))
# model.add(Dense(32, activation='relu'))
# model.add(Dropout(0.3))                    # 0.3 = 30퍼센트 빼고 훈련할거에요
# model.add(Dense(16, activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(16, activation='relu'))
# model.add(Dropout(0.1))

# model.add(Dense(1))

# model.summary()

# 함수형
input1 = Input(shape=(13,))
dense1 = Dense(32, name='ys1')(input1)
drop1 = Dropout(0.3)(dense1)
dense2 = Dense(32, name='ys2')(drop1)
drop2 = Dropout(0.3)(dense2)
dense3 = Dense(16, name ='ys3')(drop2)
drop3 = Dropout(0.2)(dense3)
dense4 = Dense(16, name='ys4')(drop3)
drop3 = Dropout(0.2)(dense4)
output1 = Dense(1)(drop3)
model = Model(inputs=input1, outputs=output1) 

model.summary()

# model.save("./_save/keras28/keras28_1_save_model.h5")   # 상대경로
# model.save("c:/AI5/_save/keras32/keras32_1_save_model.h5")   # 절대경로


# 그 모델의 가장 성능이 좋은 지점을 저장한다.

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='val_loss', mode='min',
                   patience=10, verbose=1,
                   restore_best_weights=True
                   )
import datetime
date = datetime.datetime.now()       # 현재시간 저장
print(date)         # 2024-07-26 16:50:37.570567
print(type(date))   # <class 'datetime.datetime'>
date = date.strftime("%m%d_%H%M")
print(date)
print(type(date)) # <class 'str'>


path ='./_save/keras32_mcp2/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'    # 1000-0.7777.hdf5    { } = dictionary, 키와 밸류  d는 정수, f는 소수
filepath = "".join([path, 'k32_', date, '_', filename])      # 파일위치와 이름을 에포와 발로스로 나타내준다
# 생성 예 : ""./_save/keras29_mcp/k29_1000-0.7777.hdf5"
##################### MCP 세이브 파일명 만들기 끝 ###########################

mcp = ModelCheckpoint(
    monitor='val_loss',
    mode='auto',
    verbose = 1,
    save_best_only=True,
    filepath = filepath
)


start = time.time()
hist = model.fit(x_train, y_train, epochs=100, batch_size=32,
          verbose=1,
          validation_split=0.2,
          callbacks=[es, mcp]
          )
end = time.time()

# model.save('./_save/keras29_mcp/keras29_3_save_model.h5')

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('로스 : ', loss)

y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print("r2스코어 : ", r2)
print("걸린시간 : ", round(end - start , 2),"초")


if(gpus):
    print("쥐피유 돈다!!!")
else:
    print("쥐피유 없다! xxxxx")

print("걸린시간 : ", round(end - start , 2),"초")

# 쥐피유 돈다!!!
# 걸린시간 :  5.25 초

# 쥐피유 없다! xxxxx
# 걸린시간 :  2.4 초