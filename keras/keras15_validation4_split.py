import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split

#1. 데이터
x = np.array(range(1,17))
y = np.array(range(1,17))
print(x)            # [ 1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16]

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.75, shuffle=True, random_state=123)
print(x_train, x_test)  # [ 6 10  9 12  4  2  7 16 13  3 14 15] [ 8 11  5  1]
print(y_train, y_test)  # [ 6 10  9 12  4  2  7 16 13  3 14 15] [ 8 11  5  1]

#2. 모델구성
model = Sequential()
model.add(Dense(1, input_dim=1))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=1, verbose=1,
        #   validation_data=(x_val, y_val),
          validation_split=0.3)     # 0.75 * 0.3 = 0.225

print(x_train,  x_test)

#4. 평가, 예측
print("++++++++++++++++++++")
loss = model.evaluate(x_test, y_test)
results = model.predict([18])
print("loss : ", loss)
print("[18]의 예측값 : ", results)
  
'''
loss :  0.2321098893880844
[18]의 예측값 :  [[17.28055]]
'''
