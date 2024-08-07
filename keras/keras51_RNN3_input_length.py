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
model.add(SimpleRNN(10, input_shape=(3,1)))     # units, (timesteps, feature)
# model.add(SimpleRNN(10, input_length=3, input_dim=1))
# model.add(SimpleRNN(10, input_dim=1, input_length=3))   # 가능하지만 가독성이 떨어진다.
model.add(Dense(16))
model.add(Dense(32))
model.add(Dense(64))
model.add(Dense(128))
model.add(Dense(64))
model.add(Dense(32))
model.add(Dense(16))
model.add(Dense(1))

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
