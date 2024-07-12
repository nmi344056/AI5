from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

#1. 데이터
x = np.array([1,2,3,4,5,6])
y = np.array([1,2,3,5,4,6])

#2. 모델구성
model = Sequential()
model.add(Dense(3, input_dim=1))
model.add(Dense(6))
model.add(Dense(12))
model.add(Dense(25))
model.add(Dense(12))
model.add(Dense(6))
model.add(Dense(1, input_dim=6))

#3. 컴파일, 훈련
epochs = 300
model.compile(loss='mse', optimizer='adam')
model.fit(x,y,epochs=epochs, batch_size=1)

#4. 평가, 예측
loss = model.evaluate(x, y)
results = model.predict([[7]])
print('로스 :', loss)
print('[]')