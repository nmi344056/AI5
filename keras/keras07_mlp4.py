# x 1개인데 y 3개
import numpy as np
from keras.models import Sequential
from keras.layers import Dense

#1. 데이터
x = np.array([range(10)])

y = np.array([[1,2,3,4,5,6,7,8,9,10],
              [10,9,8,7,6,5,4,3,2,1],       # x 1개, y 3개
              [9,8,7,6,5,4,3,2,1,0]])

print(x.shape)  #(3, 10)
print(y.shape)  #(2, 10)

x = x.T
y = np.transpose(y)
print(x.shape)  #(10, 3)
print(y.shape)  #(10, 2)

#2. 모델
#[실습] 맹그러봐
# x_predict = [10, 31, 211]
model = Sequential()
model.add(Dense(6, input_dim=1))
model.add(Dense(15))
model.add(Dense(30))
model.add(Dense(10))
model.add(Dense(3))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x,y,epochs=500, batch_size=1)

#4. 평가, 예측
loss = model.evaluate(x,y)
results = model.predict([10])
print('로스 :', loss)
print('[10, 31, 211]의 예측값', results)

# 로스 : 0.0012576236622408032
# [10, 31, 211]의 예측값 [[10.9811945  -0.06164578 -1.0847208 ]]