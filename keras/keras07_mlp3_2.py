import numpy as np
from keras.models import Sequential
from keras.layers import Dense

#1. 데이터
x = np.array([range(10), range(21, 31), range(201, 211)])

y = np.array([[1,2,3,4,5,6,7,8,9,10],
              [10,9,8,7,6,5,4,3,2,1],       # x 3개, y 3개
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
model.add(Dense(6, input_dim=3))
model.add(Dense(15))
model.add(Dense(30))
model.add(Dense(10))
model.add(Dense(3))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x,y,epochs=300, batch_size=1)

#4. 평가, 예측
loss = model.evaluate(x,y)
results = model.predict([[10, 31, 211]])
print('로스 :', loss)
print('[10, 31, 211]의 예측값', results)

# 로스 : 0.026733383536338806
# [10, 31, 211]의 예측값 [[10.872911    0.31128505]]
# 로스 : 0.011292724870145321
# [10, 31, 211]의 예측값 [[10.734818   -0.17855543]]
# 로스 : 0.0047580827958881855
# [10, 31, 211]의 예측값 [[10.952925    0.08582364 -0.91512   ]]
# 로스 : 0.00034042689367197454
# [10, 31, 211]의 예측값 [[ 1.1030133e+01  3.1296723e-04 -9.6925610e-01]]