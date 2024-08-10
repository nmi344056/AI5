# 56_2 copy

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM, GRU, Bidirectional
from tensorflow.keras.layers import Conv1D, Flatten
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

#1. 데이터
x = np.array([[1,2,3], [2,3,4], [3,4,5], [4,5,6],
              [5,6,7], [6,7,8], [7,8,9], [8,9,10],
              [9,10,11], [10,11,12],
              [20, 30, 40], [30, 40, 50], [40, 50, 60]])
y = np.array([4,5,6,7,8,9,10,11,12,13,50,60,70])

# [실습] Bidirectional로 만들기

print(x.shape, y.shape)     # (13, 3) (13,)

# x = x.reshape(7, 3, 1)
x = x.reshape(x.shape[0], x.shape[1], 1)
print(x.shape)              # (7, 3, 1)
# 3-D tensor with shape (batch_size, timesteps, feature).

#2. 모델 구성
model = Sequential()
model.add(Conv1D(filters=10, kernel_size=2, input_shape=(3, 1)))
model.add(Conv1D(10, 2))
model.add(Flatten())
model.add(Dense(32))
model.add(Dense(64))
model.add(Dense(128))
model.add(Dense(64))
model.add(Dense(32))
model.add(Dense(16))
model.add(Dense(1))

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
hist = model.fit(x, y, epochs=100, batch_size=1)

model.save("./_save/keras56/k56_02.h5")

#4. 평가, 예측
results = model.evaluate(x, y)
print('loss : ', results)

x_predict = np.array([50,60,70]).reshape(1,3,1)
# print(x_predict.shape)      # (1, 3, 1)
    
y_predict = model.predict(x_predict)

print('[50,60,70]의 결과 : ', y_predict)

'''
LSTM
loss :  0.21434380114078522 / [[74.5688]]

Bidirectional(LSTM
loss :  0.6971383690834045 / [[70.90632]]

Conv1D
loss :  0.16097429394721985 / [[81.55186]]

'''
