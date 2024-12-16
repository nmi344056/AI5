import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터
x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,2,3,4,5,6,7,8,9,10])

#[실습] 넘파이 리스트의 슬라이싱. 7:3으로 분할

x_train = x[:7]         # [1 2 3 4 5 6 7]
# x_train = x[0:7]      # [1 2 3 4 5 6 7]
# x_train = x[:-3]      # [1 2 3 4 5 6 7]
# x_train = x[0:-3]     # [1 2 3 4 5 6 7]
y_train = y[:7]

x_test = x[7:]          # [ 8  9 10]
# x_test = x[7:10]      # [ 8  9 10]
# x_test = x[-3:]       # [ 8  9 10]
# x_test = x[-3:10]     # [ 8  9 10]
y_test = y[7:]

print(x_train)          # [1 2 3 4 5 6 7]
print(x_test)           # [ 8  9 10]

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
epochs=200
loss :  0.14091362059116364
[11] 예측값 [11] :  [[11.539274]]
'''
