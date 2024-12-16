from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

#1. 데이터
x = np.array([1,2,3,4,5,6])
y = np.array([1,2,3,4,5,6])

#2. 모델구성
model = Sequential()
model.add(Dense(1, input_dim=1))            # y=wx+b

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=2000)

#4. 평가, 예측
loss = model.evaluate(x, y)
print("loss : ", loss)
result = model.predict([1,2,3,4,5,6,7])
print("[1,2,3,4,5,6,7]의 예측값 : ", result)

'''
로스 :  5.138560155160121e-08
[1,2,3,4,5,6,7]의 예측값 :  
[[0.99961025]
 [1.9997346 ]
 [2.999859  ]
 [3.9999835 ]
 [5.0001082 ]
 [6.0002327 ]
 [7.000357  ]]
'''
