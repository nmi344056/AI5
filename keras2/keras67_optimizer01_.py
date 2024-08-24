import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score
from sklearn.datasets import load_boston

import tensorflow as tf
tf.random.set_seed(337)     #seed를 고정하고 성능 비교
np.random.seed(337)

#1. 데이터
dataset = load_boston()
x = dataset.data
y = dataset.target

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.75, random_state=123)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델구성
model = Sequential()
model.add(Dense(10, input_dim=13))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(1))

#3. 컴파일, 훈련
from tensorflow.keras.optimizers import Adam
model.compile(loss='mse', optimizer=Adam())

model.fit(x_train, y_train, epochs=100, batch_size=32, validation_split=0.2)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test, verbose=0)
print('loss :{0}'.format(loss))

y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print('r2 :{0}'.format(r2))

'''




'''
