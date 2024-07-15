import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터
x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,2,3,4,5,6,7,8,9,10])

# x_train = x[0:7]
# x_train = x[ :7]
# x_train = [:-3]
x_train = x[0:-3]
print(x_train)       #[1 2 3 4 5 6 7]

# x_test = x[7:10]
# x_test = [7:]
x_test = x[7: ]
print(x_test)       #[ 8  9 10]

y_train = y[:7]
y_test = y[7:]


#[실습] 넘파이 리스트의 슬라이싱 !! 7:3으로 잘라라!!

#2. 모델구성

model = Sequential()
model.add(Dense(1, input_dim=1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=1, batch_size=1)

#4. 평가, 예측
print('+++++++++++++++++++++++++++++++++++++++')
loss = model.evaluate(x_test, y_test)
results = model.predict([11])
print('로스 :', loss)
print('[11]의 예측값 :', results)

# 로스 : 0.12472856044769287
# [11]의 예측값 : [[10.4888735]]