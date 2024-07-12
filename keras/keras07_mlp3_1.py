import numpy as np
from keras.models import Sequential
from keras.layers import Dense

#1. 데이터
x = np.array([range(10), range(21, 31), range(201, 211)])

y = np.array([[1,2,3,4,5,6,7,8,9,10],           # x 3개, y 2개
              [10,9,8,7,6,5,4,3,2,1]])          # input_dim=3, output_dim=2    

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
model.add(Dense(12))
model.add(Dense(24))
model.add(Dense(6))
model.add(Dense(2))

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