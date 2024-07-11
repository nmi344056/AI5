from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

#1. 데이터
x = np.array([1,2,3,4,5,6])     #데이터추가
y = np.array([1,2,3,5,4,6])

# [실습] 레이어의 깊이와 노드의 갯수를 이용해서 [6]을 맹그러
# 에포는 100으로 고정, 건들지말것!!!
# 소수 네째자리까지 맞추면 합격. 예: 6.0000 또는 5.9999

#2. 모델구현
model = Sequential()
model.add(Dense(3, input_dim =1))
model.add(Dense(6))
model.add(Dense(12))
model.add(Dense(25))
model.add(Dense(12))
model.add(Dense(6))
model.add(Dense(1, input_dim =6))


#3. 컴파일, 훈련
epochs = 1000
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs = epochs, batch_size=2)

#4. 평가, 예측
loss = model.evaluate(x, y)
print("===========================")
print("epochs = ", epochs)
print("로스 : ", loss)
result = model.predict([7])
print("7의 예측값 : ", result)

# epochs =  2000
# 로스 :  0.3284194767475128
# 7의 예측값 :  [[6.6936216]]  


import numpy as np

x1 = np.array([1,2,3])
print("x1 : ", x1.shape)