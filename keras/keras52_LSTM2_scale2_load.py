import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM, GRU
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

#1. 데이터
x = np.array([[1,2,3], [2,3,4], [3,4,5], [4,5,6],
              [5,6,7], [6,7,8], [7,8,9], [8,9,10],
              [9,10,11], [10,11,12],
              [20, 30, 40], [30, 40, 50], [40, 50, 60]])
y = np.array([4,5,6,7,8,9,10,11,12,13,50,60,70])

print(x.shape, y.shape)     # (13, 3) (13,)

# [실습] x_predict = np.array([50,60,70])   -> 80 출력


# x = x.reshape(7, 3, 1)
x = x.reshape(x.shape[0], x.shape[1], 1)
print(x.shape)              # (7, 3, 1)
# 3-D tensor with shape (batch_size, timesteps, feature).

# #2. 모델 구성
# model = Sequential()
# model.add(LSTM(10, input_shape=(3,1)))        # 3개 성능차이 있다. 골라서 사용
# model.add(Dense(16))
# model.add(Dense(32))
# model.add(Dense(64))
# model.add(Dense(128))
# model.add(Dense(64))
# model.add(Dense(32))
# model.add(Dense(16))
# model.add(Dense(1))

# # 3. 컴파일, 훈련
# model.compile(loss='mse', optimizer='adam')
# hist = model.fit(x, y, epochs=100, batch_size=10)

# model = load_model("./_save/keras52/k52_02.h5")
model = load_model("./_save/keras52/k52_02_79.11273.h5")

#4. 평가, 예측
results = model.evaluate(x, y)
print('loss : ', results)

x_predict = np.array([50,60,70]).reshape(1,3,1)
print(x_predict.shape)      # (1, 3, 1)
    
y_predict = model.predict(x_predict)

print('[50,60,70]의 결과 : ', y_predict)

'''
[실습] 80.0 만들기

loss :  3.373504638671875
(1, 3, 1)
[8,9,10]의 결과 :  [[70.99619]]

'''
