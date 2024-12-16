from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.datasets import load_diabetes

#1. 데이터
datasets = load_diabetes()
x = datasets.data
y = datasets.target

print(x)
print(y)
print(x.shape, y.shape)          # (442, 10) (442,)

#[실습]만들기 R2 성능 0.62 이상

random_state=8000
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.79, random_state=random_state)

#2. 모델구성
model = Sequential()
model.add(Dense(100, input_dim=10))
model.add(Dense(75))
model.add(Dense(50))
model.add(Dense(30))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=500, batch_size=10, validation_split=0.2)

#4.평가, 예측
print("++++++++++++++++++++")
loss = model.evaluate(x_test, y_test)
print("random_state : ", random_state)
print("loss : ", loss)

y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print("r2 sorce : ", r2)

'''
100 75 50 30 1 (0.79)
random_state :  999
loss :  2237.555908203125
r2 sorce :  0.5779882175127407
'''
