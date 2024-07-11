from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

#1. 데이터
x = np.array([1,2,3,4,5,6])
y = np.array([1,2,3,5,4,6])

"""
[실습]
레이어의 깊이와 노드의 갯수를 이용해서 최소의 loss를 만들기
epochs는 100으로 고정
loss 기준 0.32 미만
"""

#2. 모델구성
model = Sequential()
model.add(Dense(1, input_dim=1))
model.add(Dense(100))
model.add(Dense(3))
model.add(Dense(100))
model.add(Dense(3))

epochs = 100
#3. 컴파일 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=epochs)

#4. 평가, 예측
loss = model.evaluate(x, y)
print("====================")
print("로스 : ", loss)
result = model.predict([6])
print("6의 예측값 : ", result)

"""
5. 결과값 기록
3 100 3 100 3
로스 :  0.32666540145874023
6의 예측값 :  [[5.835899  5.7816343 5.7988706]]
"""
