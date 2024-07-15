import sklearn as sk
print(sk.__version__)
from sklearn.datasets import fetch_california_housing
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터
datasets = fetch_california_housing()
x = datasets.data
y = datasets.target

print(x)
print(y)
print(x.shape, y.shape)     #(20640, 8) (20640,)

#[실습] 맹그러
# R2 0.59 이상

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7,
                                                    random_state=9000)
print('x_train :', x_train)
print('x_test :', x_test)
print('y_train :', y_train)
print('y_test :', y_test)

#2. 모델구성
model = Sequential()
model.add(Dense(10, input_dim=8))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=50, batch_size=16)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('로스 : ', loss)

y_predict = model.predict(x_test)
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print("r2스코어 : ", r2)
