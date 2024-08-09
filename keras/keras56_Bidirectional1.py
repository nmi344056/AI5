# 51_1 copy

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM, GRU, Bidirectional

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
# model.add(Bidirectional(10, input_shape=(3,1)))		# 타입에러 : 미싱~~~

# model.add(SimpleRNN(10, input_shape=(3,1)))                           # Param : 120
# model.add(Bidirectional(SimpleRNN(units=10), input_shape=(3,1)))      # Param : 240

# model.add(GRU(10, input_shape=(3,1)))                                 # Param : 390
# model.add(Bidirectional(GRU(units=10), input_shape=(3,1)))            # Param : 780

# model.add(LSTM(10, input_shape=(3,1)))                                # Param : 480
model.add(Bidirectional(LSTM(units=10), input_shape=(3,1)))             # Param : 960

model.add(Dense(7))
model.add(Dense(1))

model.summary()
'''
 Layer (type)                Output Shape              Param #
=================================================================
 bidirectional (Bidirectiona  (None, 20)               240
 l)
 dense (Dense)               (None, 7)                 147
 dense_1 (Dense)             (None, 1)                 8

=================================================================
Total params: 395
Trainable params: 395
Non-trainable params: 0
'''

#[실습] SimpleRNN을 GRU, LSTM로 바꿔서 실행, *2가 되는지 확인

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=100)

#4. 평가, 예측
results = model.evaluate(x, y)
print('loss : ', results)

x_predict = np.array([8,9,10]).reshape(1,3,1)
print(x_predict.shape)      # (1, 3, 1)
    
y_predict = model.predict(x_predict)

print('[8,9,10]의 결과 : ', y_predict)

'''
[실습] 11.0 만들기
loss :  0.014437743462622166
[8,9,10]의 결과 :  [[10.5046215]]

'''
