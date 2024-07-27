import numpy as np
import pandas as pd
import time

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.datasets import load_wine

#1. 데이터
datasets = load_wine()

x = datasets.data
y = datasets.target
print(x.shape, y.shape)     # (178, 13) (178,)

print(y)
print(np.unique(y, return_counts=True))
# (array([0, 1, 2]), array([59, 71, 48], dtype=int64))

from tensorflow.keras.utils import to_categorical
y_ohe1 = to_categorical(y)
print(y_ohe1)
print(y_ohe1.shape)         # (178, 3)

x_train, x_test, y_train, y_test = train_test_split(x, y_ohe1, train_size=0.9, random_state=666)

print(x_train.shape, x_test.shape)      # (142, 13) (36, 13)
print(y_train.shape, y_test.shape)      # (142, 3) (36, 3)

#2. 모델구성
model = Sequential()
model.add(Dense(32, activation='relu', input_dim=13))
model.add(Dense(16, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(3, activation='softmax'))

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=100, batch_size=1, validation_split=0.2)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test, verbose=1)
print("loss : ", loss[0])
print("accuracy : ", round(loss[1], 3))

y_predict = model.predict(x_test)
print(y_predict[:20])
y_predict = np.round(y_predict)
print(y_predict[:20])

accuracy_score = accuracy_score(y_test, y_predict)
print("acc score : ", accuracy_score)

'''
acc score :  0.6388888888888888
acc score :  1.0
'''
