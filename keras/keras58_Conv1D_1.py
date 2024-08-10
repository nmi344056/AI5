# 56_1 copy

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM, GRU, Bidirectional
from tensorflow.keras.layers import Conv1D, Flatten

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
model.add(Conv1D(filters=10, kernel_size=2, input_shape=(3, 1)))
model.add(Conv1D(10, 2))
model.add(Flatten())
model.add(Dense(7))
model.add(Dense(1))

model.summary()
'''
 Layer (type)                Output Shape              Param #
=================================================================
 conv1d (Conv1D)             (None, 2, 10)             30
 flatten (Flatten)           (None, 20)                0
 dense (Dense)               (None, 7)                 147
 dense_1 (Dense)             (None, 1)                 8
=================================================================
Total params: 185
Trainable params: 185
Non-trainable params: 0
'''

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=2000)

#4. 평가, 예측
results = model.evaluate(x, y)
print('loss : ', results)

x_predict = np.array([8,9,10]).reshape(1,3,1)
# print(x_predict.shape)      # (1, 3, 1)
    
y_predict = model.predict(x_predict)

print('[8,9,10]의 결과 : ', y_predict)

'''
epochs=2000
loss :  1.0556634640881621e-13
[8,9,10]의 결과 :  [[11.000001]]

'''














