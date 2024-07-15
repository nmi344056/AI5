import sklearn as sk
print (sk.__version__)      # 0.24.2 / 1.4.2

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.datasets import load_boston

#1. 데이터
dataset = load_boston()
# print(dataset)
x = dataset.data
y = dataset.target

random_state=555
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=random_state)
print(x)
print(x.shape)             # (506, 13)
print(y)
print(y.shape)             # (506,)

#2. 모델구성
model = Sequential()
model.add(Dense(3, input_dim=13))
model.add(Dense(3))
model.add(Dense(3))
model.add(Dense(3))
model.add(Dense(3))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=500, batch_size=10)

#4. 평가, 예측
print("++++++++++++++++++++")
loss = model.evaluate(x_test, y_test)
print("random_state : ", random_state)
print("loss : ", loss)

y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print("r2 score : ", r2)

'''
0.8 이상
100 30 500 30 100 30 1
random_state :  555
loss :  21.663753509521484
r2 score :  0.7484352588268719
'''
