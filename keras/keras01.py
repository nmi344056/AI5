import tensorflow as tf
print(tf.__version__)

from tensorflow.keras.models import Sequential  # tensorflow 생략 가능
from tensorflow.keras.layers import Dense
import numpy as np

#1. 데이터
x = np.array([1,2,3,])
y = np.array([1,2,3])

#2. 모델구성
model = Sequential()                            # 순차적으로 연산하는 모델
model.add(Dense(1, input_dim=1))                # input 1개에 output 1개

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')     # 컴퓨터가 알아듣게 컴파일한다.
model.fit(x, y, epochs=100)                     # 훈련 횟수를 높일수록 예측값 정확

#4. 평가, 예측
result = model.predict(np.array([4]))
print("[4]의 예측값 : ", result)                  # [[2.676833]]
