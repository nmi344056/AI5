# keras08_1 copy

# x train test.

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터
# x = np.array([1,2,3,4,5,6,7,8,9,10])
# y = np.array([1,2,3,4,5,6,7,8,9,10])

x_train = np.array([1,2,3,4,5,6,7])
y_train = np.array([1,2,3,4,5,6,7])

x_test = np.array([8,9,10])
y_test = np.array([8,9,10])

#2. 모델구성

model = Sequential()
model.add(Dense(1, input_dim=1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=500, batch_size=1,
          verbose=3          #verbode=0 바로출력, 1 디폴트, 2 진행바 삭제, 3이상 epochs만 출력
          )
# verbose=0 : 침묵
# verbose=1 : 디포트
# verbose=2 : 프로그래스바 삭제
# verbose=3이상 나머지 : 에포만 나온다.


#4. 평가, 예측
print('+++++++++++++++++++++++++++++++++++++++')
loss = model.evaluate(x_test, y_test)
results = model.predict([11])
print('로스 :', loss)
print('[11]의 예측값 :', results)

# 로스 : 0.00013008674432057887
# [11]의 예측값 : [[11.016525]]


# 7/7 [==============================] - 0s 332us/step - loss: 9.6975e-06
# +++++++++++++++++++++++++++++++++++++++
# 1/1 [==============================] - 0s 33ms/step - loss: 3.8741e-05
# 로스 : 3.87412728741765e-05
# [11]의 예측값 : [[10.991015]]