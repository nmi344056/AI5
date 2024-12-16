import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split

#1. 데이터
x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,2,3,4,7,5,7,8,6,10])

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, shuffle=True, random_state=23)

#2. 모델구성
model = Sequential()
model.add(Dense(1, input_dim=1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=1)

#4. 평가, 예측
print("++++++++++++++++++++")
loss = model.evaluate(x_test, y_test)
result = model.predict(x)

print("loss : ", loss)
print("x의 예측값 : ", result)

import matplotlib.pyplot as plt
plt.scatter(x, y)
plt.plot(x, result, color='red')
plt.show()

'''
loss :  3.7444517612457275
x의 예측값 :  
[[ 1.2781501]
 [ 2.2591982]
 [ 3.2402463]
 [ 4.2212944]
 [ 5.2023425]
 [ 6.1833906]
 [ 7.1644387]
 [ 8.145487 ]
 [ 9.126535 ]
 [10.107583 ]]
'''
