from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

#1. 데이터
x = np.array([1,2,3,4,5])
y = np.array([1,2,4,3,5])

#[실습] 직접작성

#2. 모델 구성
model = Sequential()
model.add(Dense(1, input_dim=1))

epochs = 2000            #튜닝
#3. 컴파일 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=epochs)

#4. 평가, 예측
loss = model.evaluate(x, y)
print("====================")
print("epochs : ",epochs)
print("loss : ", loss)
result = model.predict([6])
print("[6]의 예측값 : ", result)

"""
5. 결과값 기록
epochs :  1000
로스 :  0.38007014989852905
[6]의 예측값 :  [[5.713788]]

"""
