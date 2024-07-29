# keras19_EarlyStopping2_california copy

import numpy as np
import sklearn as sk
from sklearn.datasets import fetch_california_housing
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input
import time

#1. 데이터
datasets = fetch_california_housing()
x = datasets.data
y = datasets.target

print(x)
print(y)
print(x.shape, y.shape)

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.9,
                                                    random_state=3)
print('x_train :', x_train)
print('x_test :', x_test)
print('y_train :', y_train)
print('y_test :', y_test)


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

# #2. 모델구성
# model = Sequential()
# model.add(Dense(16,activation='relu', input_dim=8))
# model.add(Dropout(0.3))
# model.add(Dense(32,activation='relu'))
# model.add(Dropout(0.3))
# model.add(Dense(64,activation='relu'))
# model.add(Dropout(0.3))
# model.add(Dense(64,activation='relu'))
# model.add(Dropout(0.3))
# model.add(Dense(32,activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(16,activation='relu'))
# model.add(Dense(8,activation='relu'))
# model.add(Dense(4,activation='relu'))
# model.add(Dense(1))

#2-2. 모델구성(함수형)
input1 = Input(shape=(8,))
dense1 = Dense(16, activation='relu', name='ys1')(input1)
drop1 = Dropout(0.3)(dense1)
dense2 = Dense(32, activation='relu', name='ys2')(drop1)
drop2 = Dropout(0.3)(dense2)
dense3 = Dense(64,activation='relu', name='ys3')(drop2)
drop3 = Dropout(0.3)(dense3)
dense4 = Dense(64,activation='relu', name='ys4')(drop3)
drop4 = Dropout(0.3)(dense4)
dense5 = Dense(32,activation='relu', name='ys5')(drop4)
drop5 = Dropout(0.2)(dense5)
dense6 = Dense(16,activation='relu', name='ys6')(drop5)
dense7=Dense(8,activation='relu', name='ys7')(dense6)
dense8=Dense(4,activation='relu',name='ys8')(dense7)
output1 = Dense(1)(dense8)
model = Model(inputs=input1, outputs=output1)
model.summary()




#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
start = time.time()

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(
    monitor = 'val_loss',
    mode = 'min',
    patience=10,
    restore_best_weights=True,
)
import datetime
date = datetime.datetime.now()       # 현재시간 저장
print(date)         # 2024-07-26 16:50:37.570567
print(type(date))   # <class 'datetime.datetime'>
date = date.strftime("%m%d_%H%M")
print(date)
print(type(date)) # <class 'str'>


path ='C:/AI5/_save/keras32_mcp/02_california/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'    # 1000-0.7777.hdf5    { } = dictionary, 키와 밸류  d는 정수, f는 소수
filepath = "".join([path, 'k32_02', date, '_', filename])      # 파일위치와 이름을 에포와 발로스로 나타내준다
# 생성 예 : ""./_save/keras29_mcp/k29_1000-0.7777.hdf5"
##################### MCP 세이브 파일명 만들기 끝 ###########################

mcp = ModelCheckpoint(
    monitor='val_loss',
    mode='auto',
    verbose = 1,
    save_best_only=True,
    filepath = filepath
)


hist = model.fit(x_train, y_train,
          validation_split=0.25, epochs=500, batch_size=100,
          callbacks=[es, mcp])
end = time.time()

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('로스 : ', loss)

y_predict = model.predict(x_test)
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print("r2스코어 : ", r2)
print("걸린시간 :", round(end-start, 2),"초")     # round 2 는 반올림자리

# 로스 :  0.5070632100105286
# r2스코어 :  0.623687375535644
# 걸린시간 : 3.29 초

# Dropout 적용후
# 로스 :  0.48401007056236267
# r2스코어 :  0.640796079121912
# 걸린시간 : 17.2 초