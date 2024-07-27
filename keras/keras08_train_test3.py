import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터
x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,2,3,4,5,6,7,8,9,10])

#[검색] train과 test를 섞어서 7:3으로 분할 (사이킷런)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, shuffle=True, random_state=123)

print("x_train : ", x_train)    # x_train :  [ 6  9  4  2  7 10  3]
print("x_test : ", x_test)      # x_test :  [5 1 8]
print("y_train : ", y_train)    # y_train :  [ 6  9  4  2  7 10  3]
print("y_test : ", y_test)      # y_test :  [5 1 8]

#2. 모델구성
model = Sequential()
model.add(Dense(1, input_dim=1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=200, batch_size=1)

#4. 평가, 예측
print("++++++++++++++++++++")
loss = model.evaluate(x_test,y_test)
result = model.predict([11])
print("loss : ", loss)
print("[11] 예측값 [11] : ", result)

'''
random_state=23 / epochs=200
loss :  0.029091497883200645
[11] 예측값 [11] :  [[11.326617]]
'''
