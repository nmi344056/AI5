import tensorflow as tf
print(tf.__version__)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

#1. 데이터
x = np.array([1,2,3,])
y = np.array([1,2,3])

#2. 모델구성
model = Sequential()
model.add(Dense(1, input_dim=1)) #인풋 한덩어리, 아웃풋 한덩어리

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam') #컴퓨터가 알아먹게 컴파일한다.
model.fit(x, y, epochs=100)

#4. 평가, 예측
result = model.predict(np.array([4]))
print("4의 예측값 : ", result)
