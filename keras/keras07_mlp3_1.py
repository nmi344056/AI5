import numpy as np
from keras.models import Sequential
from keras.layers import Dense

#1. 데이터
x = np.array([range(10), range(21, 31), range(201, 211)])
y = np.array([[1,2,3,4,5,6,7,8,9,10],[10,9,8,7,6,5,4,3,2,1]])

print(x.shape)          # (3, 10)
print(y.shape)          # (2, 10)

x = x.T
y = np.transpose(y)

print(x.shape)          # (10, 3)
print(y.shape)          # (10, 2)

# [실습] x_predict = [10, 31, 211]

#2. 모델
model = Sequential()
model.add(Dense(10, input_dim=3))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(2))     # output_dim=2

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x,y, epochs=100, batch_size=1)

#4. 평가, 예측
loss = model.evaluate(x,y)
result = model.predict([[10, 31, 211]])
print("loss : ", loss)
print("11 0 예측값 : ", result)

"""
loss :  0.16846707463264465
[10, 31, 211] 예측값 :  [[11.028464   0.7577088]]
"""
