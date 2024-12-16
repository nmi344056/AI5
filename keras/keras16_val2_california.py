from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.datasets import fetch_california_housing

#1. 데이터
datasets = fetch_california_housing()
x = datasets.data
y = datasets.target

print(x)
print(y)
print(x.shape, y.shape)         # (20640, 8) (20640,)

#[실습] 만들기 R2 성능 0.59 이상
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=123)

#2. 모델구성
model = Sequential()
model.add(Dense(60, input_dim=8))
model.add(Dense(30))
model.add(Dense(15))
model.add(Dense(7))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=500, batch_size=100, validation_split=0.2)

#4. 평가,예측
print("++++++++++++++++++++")
loss = model.evaluate(x_test, y_test)
print("loss : ", loss)

y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print("r2 score : ", r2)

'''
3 4 5 3 1
train_size=0.7, random_state=123 / 
loss :  0.588417649269104
r2 score :  0.5550008546448328
'''
