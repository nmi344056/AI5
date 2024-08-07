# 51_2 copy

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM, GRU

#1. 데이터
datasets = np.array([1,2,3,4,5,6,7,8,9,10])

x = np.array([[1,2,3],
              [2,3,4],
              [3,4,5],
              [4,5,6],
              [5,6,7],
              [6,7,8],
              [7,8,9]])

y = np.array([4,5,6,7,8,9,10])

print(x.shape, y.shape)     # (7, 3) (7,)

# x = x.reshape(7, 3, 1)
x = x.reshape(x.shape[0], x.shape[1], 1)
print(x.shape)              # (7, 3, 1)
# 3-D tensor with shape (batch_size, timesteps, feature).

#2. 모델 구성
model = Sequential()
# model.add(LSTM(10, input_shape=(3,1)))     # units, (timesteps, feature)
model.add(GRU(10, input_shape=(3,1)))
model.add(Dense(7))
model.add(Dense(1))

model.summary()
'''
LSTM
 Layer (type)                Output Shape              Param #
=================================================================
 lstm (LSTM)                 (None, 10)                480
 dense (Dense)               (None, 7)                 77
 dense_1 (Dense)             (None, 1)                 8
=================================================================
Total params: 565
Trainable params: 565
Non-trainable params: 0

[검색] LSTM의 Param이 왜 480인지 찾기
입력 게이트 (Input Gate), 망각 게이트 (Forget Gate), 출력 게이트 (Output Gate), Cell State로 *4

GRU
 Layer (type)                Output Shape              Param #
=================================================================
 gru (GRU)                   (None, 10)                390
 dense (Dense)               (None, 7)                 77
 dense_1 (Dense)             (None, 1)                 8
=================================================================
Total params: 475
Trainable params: 475
Non-trainable params: 0

[검색] GRU의 Param이 왜 390인지 찾기
옛날엔 딱 3배였는제 지금은 달라짐

'''
