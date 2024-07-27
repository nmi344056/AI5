import numpy as np
import pandas as pd
import time

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score
from sklearn.datasets import load_breast_cancer

#1. 데이터
datasets = load_breast_cancer()
# print(datasets)
print(datasets.DESCR)
print(datasets.feature_names)

x = datasets.data
y = datasets.target
print(x.shape, y.shape)                     # (569, 30) (569,)

# [실습] 0과 1의 갯수가 몇개인지 찾기

print(np.unique(y))                         # [0 1] 꼭 확인
print(np.unique(y, return_counts=True))     # (array([0, 1]), array([212, 357], dtype=int64))

print(type(x))                              # <class 'numpy.ndarray'>
print(pd.DataFrame(y).value_counts())
# 1    357
# 0    212

# print(y.value_counts())                   # AttributeError: 'numpy.ndarray' object has no attribute 'value_counts'

print(pd.Series(y))
print("++++++++++++++++++++")
print(pd.Series(y).value_counts())
print(pd.value_counts(y))


x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=555)

print(x_train.shape, y_train.shape)         # (455, 30) (455,)
print(x_test.shape, y_test.shape)           # (114, 30) (114,)

#2. 모델구성
model = Sequential()
model.add(Dense(32, activation='relu', input_dim=30))
model.add(Dense(16, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

#3. 컴파일, 훈련
# model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

start_time = time.time()
# es = EarlyStopping(monitor='val_loss', mode='min', patience=10, restore_besr_weights=True)
hist = model.fit(x_train, y_train, epochs=100, batch_size=8, validation_split=0.2)      # , callbacks=[es]
end_time = time.time()

#4. 평가, 예측
loss = model.evaluate(x_test, y_test, verbose=1)
print("loss : ", loss[0])
print("accuracy : ", round(loss[1], 3))

y_predict = model.predict(x_test)
print(y_predict[:20])               # y' 결과
y_predict = np.round(y_predict)
print(y_predict[:20])               # y' 반올림 결과

accuracy_score = accuracy_score(y_test, y_predict)
print("acc score : ", accuracy_score)
print("time : ", round(end_time - start_time, 2), "초")

'''
32 16 16 16 16 1
train_size=0.8, random_state=555 / epochs=100, batch_size=8, validation_split=0.2 / verbose=1
loss :  0.09497827291488647
accuracy :  0.886
++++++++++++++++++++
 [1.]
 [0.]
 [0.]
 [1.]]
acc score :  0.956140350877193
binary_crossentropy
acc score :  0.9035087719298246
'''
