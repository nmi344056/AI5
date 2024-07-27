import sklearn as sk
print (sk.__version__)      # 0.24.2 / 1.4.2

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.datasets import load_boston
import numpy as np

#1. 데이터
dataset = load_boston()

print(dataset)
print(dataset.DESCR)
print(dataset.feature_names)
# ['CRIM' 'ZN' 'INDUS' 'CHAS' 'NOX' 'RM' 'AGE' 'DIS' 'RAD' 'TAX' 'PTRATIO' 'B' 'LSTAT']

x = dataset.data
y = dataset.target

print(x)
print(x.shape)             # (506, 13)
print(y)
print(y.shape)             # (506,)

# [실습] train_size : 0.7~0.9 ,  r2 : 0.8 이상
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=555)

#2. 모델구성
model = Sequential()
model.add(Dense(100, input_dim=13))
model.add(Dense(50))
model.add(Dense(25))
model.add(Dense(10))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=500, batch_size=10)

#4. 평가, 예측
print("++++++++++++++++++++")
loss = model.evaluate(x_test, y_test)
print("loss : ", loss)

y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print("r2 score : ", r2)

'''
100 50 25 10 1
train_size=0.8, random_state=555 / epochs=500, batch_size=10
loss :  19.516820907592773
r2 score :  0.7522954574324735
'''
