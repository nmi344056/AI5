from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

#1. 데이터
x = np.array([1,2,3,4,5])
y = np.array([1,2,3,4,5])

"""
[실습]
레이어의 깊이와 노드의 개수를 이용해서 [6]을 만들기
epochs는 100으로 고정
소수 넷째 자리까지 맞추면 합격 (예 : 6.0000 또는 5.9999)
"""

#2. 모델구성
model = Sequential()
model.add(Dense(3, input_dim=1))    # input_dim=input layer의 노드(열) 수
model.add(Dense(4, input_dim=3))
model.add(Dense(5, input_dim=4))
model.add(Dense(4, input_dim=5))
model.add(Dense(1, input_dim=4))

epochs = 100
#3. 컴파일 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=epochs)

#4. 평가, 예측
loss = model.evaluate(x, y)
print("====================")
print("loss : ", loss)
result = model.predict([6])
print("[6]의 예측값 : ", result)

"""
5. 결과값 기록
3 4 5 4 1
loss :  1.4377397405951342e-07
[6]의 예측값 :  [[6.0005965]]
====================
3 4 3 1
loss :  2.49858771894651e-06
[6]의 예측값 :  [[5.9993324]]
"""
