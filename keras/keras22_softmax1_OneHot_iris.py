import numpy as np
import pandas as pd
import time
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.datasets import load_iris

#1. 데이터
datasets = load_iris()
# print(datasets)
# print(datasets.DESCR)
# print(datasets.feature_names)

x = datasets.data
y = datasets.target
print(x.shape, y.shape)     # (150, 4) (150,)

print(y)
print(np.unique(y, return_counts=True))
# (array([0, 1, 2]), array([50, 50, 50], dtype=int64))

print(pd.value_counts(y))
# 0    50
# 1    50
# 2    50

# from tensorflow.keras.utils import to_categorical
# y_ohe1 = to_categorical(y)
# print(y_ohe1)
# print(y_ohe1.shape)                 # (150, 3)

# y_ohe2 = pd.get_dummies(y[0])       # pandas
# y_ohe2 = pd.get_dummies(y)          # pandas
# print(y_ohe2)
# print(y_ohe2.shape)                 # (150, 3)

# print("==============================")
# from sklearn.preprocessing import OneHotEncoder
# y_ohe3 = y.reshape(-1, 1)           # (150, 1)
# ohe = OneHotEncoder(sparse=False)   # True가 default
# # y_ohe3 = ohe.fit_transform(y_ohe3)
# ohe.fit(y_ohe3)
# y_ohe3 = ohe.transform(y_ohe3)
# print(y_ohe3)

# Reshape your data either using array.reshape(-1, 1) if your data has a single feature or array.reshape(1, -1) if it contains a single sample.
# y는 벡터형태 (150,) / (-1(통상 데이터의 끝), 1)=전체 데이터=(150,1) 으로 바꿔라. 메트릭스(행렬)형태로 받겠다. reshape 조건 : 값이 바뀌면 안된다. 순서가 바뀌면 안된다.

########## [실습] 만들기 ##########
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.9, random_state=666)

print(x_train.shape, x_test.shape)      # (120, 4) (30, 4)
print(y_train.shape, y_test.shape)      # (120, 3) (30, 3)

print(pd.value_counts(y_train))
# 0    46
# 2    46
# 1    43

# #2. 모델구성
# model = Sequential()
# model.add(Dense(32, activation='relu', input_dim=4))
# model.add(Dense(16, activation='relu'))
# model.add(Dense(16, activation='relu'))
# model.add(Dense(16, activation='relu'))
# model.add(Dense(16, activation='relu'))
# model.add(Dense(16, activation='relu'))
# model.add(Dense(16, activation='relu'))
# model.add(Dense(16, activation='relu'))
# model.add(Dense(16, activation='relu'))
# model.add(Dense(16, activation='relu'))
# model.add(Dense(3, activation='softmax'))

# #3. 컴파일, 훈련
# # model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
# # model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# start_time = time.time()
# # es = EarlyStopping(monitor='val_loss', mode='min', patience=10, restore_besr_weights=True)
# hist = model.fit(x_train, y_train, epochs=100, batch_size=8, validation_split=0.2)      # , callbacks=[es]
# end_time = time.time()

# #4. 평가, 예측
# loss = model.evaluate(x_test, y_test, verbose=1)
# print("loss : ", loss[0])
# print("accuracy : ", round(loss[1], 3))

# y_predict = model.predict(x_test)
# print(y_predict[:20])               # y' 결과
# y_predict = np.round(y_predict)
# print(y_predict[:20])               # y' 반올림 결과

# accuracy_score = accuracy_score(y_test, y_predict)
# print("acc score : ", accuracy_score)
# print("time : ", round(end_time - start_time, 2), "초")

# '''
#  [ 2.]
#  [-0.]
#  [-0.]
#  [ 2.]
#  [-0.]]
# acc score :  0.9666666666666667

#  [1. 0. 0.]
#  [0. 1. 0.]
#  [1. 0. 0.]]
# acc score :  0.9666666666666667
# '''
