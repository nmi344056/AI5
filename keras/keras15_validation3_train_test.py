import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split

#1. 데이터
x = np.array(range(1,17))
y = np.array(range(1,17))
print(x)            # [ 1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16]

# train_test_split로만 자르기

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.75, shuffle=False)
print(x_train)      # [ 1  2  3  4  5  6  7  8  9 10 11 12]
print(x_test)       # [13 14 15 16]

x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size=0.7, shuffle=False)
print(x_train)      # [1 2 3 4 5 6 7 8]
print(x_val)        # [ 9 10 11 12]
print(x_test)       # [13 14 15 16]

#2. 모델구성
model = Sequential()
model.add(Dense(1, input_dim=1))
model.add(Dense(10))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=1000, batch_size=1, verbose=1, validation_data=(x_val, y_val))

#4. 평가, 예측
print("++++++++++++++++++++")
loss = model.evaluate(x_test,y_test)
result = model.predict([17])
print("loss : ", loss)
print("[17]의 예측값 : ", result)

'''
loss :  0.0
[17]의 예측값 :  [[17.]]
r2 sorce :  0.9999999999990905
'''
