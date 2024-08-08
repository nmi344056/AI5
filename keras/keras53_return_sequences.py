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

# [실습] LSTM을 2개 이상 넣은 모델 구성

x = x.reshape(x.shape[0], x.shape[1], 1)
print(x.shape)              # (7, 3, 1)
# 3-D tensor with shape (batch_size, timesteps, feature).

#2. 모델 구성
model = Sequential()
model.add(LSTM(16, input_shape=(3,1), return_sequences=True))
model.add(LSTM(9))
model.add(Dense(7))
model.add(Dense(1))

model.summary()
'''
 Layer (type)                Output Shape              Param #
=================================================================
 lstm (LSTM)                 (None, 3, 16)             1152
 lstm_1 (LSTM)               (None, 9)                 936
 dense (Dense)               (None, 7)                 70
 dense_1 (Dense)             (None, 1)                 8
=================================================================
Total params: 2,166
Trainable params: 2,166
Non-trainable params: 0
'''
