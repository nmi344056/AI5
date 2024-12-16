import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
# import xgboost as XGBClassfier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

import tensorflow as tf
tf.random.set_seed(33)

#1. 데이터
datasets = load_wine()
x = datasets.data
y = datasets['target']

print(x.shape, y.shape) # (178, 13) (178,)
print(np.unique(y, return_counts=True))
# (array([0, 1, 2]), array([59, 71, 48], dtype=int64))
print(pd.value_counts(y))
# 1    71
# 0    59
# 2    48
print(y)
'''
[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2
 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2]
 '''

x = x[:-39]
y = y[:-39]

print(np.unique(y, return_counts=True))
# (array([0, 1, 2]), array([59, 71,  8], dtype=int64))
print(y)
'''
[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2]
 불균형 데이터가 됐다.
 '''

# from tensorflow.keras.utils import to_categorical
# y = to_categorical(y)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.75, stratify=y, shuffle=True, random_state=123)

'''
#2. 모델 구성
model = Sequential()
model.add(Dense(10, input_shape=(13,)))
model.add(Dense(3, activation='softmax'))

#3. 컴파일, 훈련
# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])   # onehot 없어도 된다.
model.fit(x_train, y_train, epochs=100, validation_split=0.2)

#4. 평가, 예측
results = model.evaluate(x_test, y_test)
print('loss :', results[0])
print('accuracy :', results[1])

# 지표 : f1_score
y_predict = model.predict(x_test)
# print(y_predict)    #  [9.92437780e-01 4.84594796e-03 2.71617435e-03]], 3개의 값이 나오니까 argmax 필요

y_predict = np.argmax(y_predict, axis=1)
# print(y_predict)    # [1 0 0 0 1 1 0 0 1 0 1 0 0 0 1 1 0 0 0 0 0 0 1 0 1 0 1 1 1 1 0 1 0 1 0]

acc = accuracy_score(y_test, y_predict)
f1 = f1_score(y_test, y_predict, average='macro')
print('acc :', acc)
print('f1 :', f1)

# loss : 0.6588544845581055
# accuracy : 0.8571428656578064
# acc : 0.8571428571428571
# f1 : 0.5970961887477314

'''

########## ROS 적용 ##########
# cmd > pip install imblearn
from imblearn.over_sampling import SMOTE, RandomOverSampler
import sklearn as sk
print(sk.__version__)   # 1.5.1

print('증폭전:', np.unique(y_train, return_counts=True))
# 증폭전: (array([0, 1, 2]), array([44, 53,  6], dtype=int64))

# smote = SMOTE(random_state=7777)
ros = RandomOverSampler(random_state=7777)
x_train, y_train = ros.fit_resample(x_train, y_train)
print('증폭후:', np.unique(y_train, return_counts=True))
# 증폭후: (array([0, 1, 2]), array([53, 53, 53], dtype=int64))

########## ROS 적용 끝 ##########

#2. 모델 구성
model = Sequential()
model.add(Dense(10, input_shape=(13,)))
model.add(Dense(3, activation='softmax'))

#3. 컴파일, 훈련
# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])   # onehot 없어도 된다.
model.fit(x_train, y_train, epochs=100, validation_split=0.2)

#4. 평가, 예측
results = model.evaluate(x_test, y_test)
print('loss :', results[0])
print('accuracy :', results[1])

# 지표 : f1_score
y_predict = model.predict(x_test)
# print(y_predict)    #  [9.92437780e-01 4.84594796e-03 2.71617435e-03]], 3개의 값이 나오니까 argmax 필요

y_predict = np.argmax(y_predict, axis=1)
# print(y_predict)    # [1 0 0 0 1 1 0 0 1 0 1 0 0 0 1 1 0 0 0 0 0 0 1 0 1 0 1 1 1 1 0 1 0 1 0]

acc = accuracy_score(y_test, y_predict)
f1 = f1_score(y_test, y_predict, average='macro')
print('acc :', acc)
print('f1 :', f1)

'''
loss : 0.6588544845581055
accuracy : 0.8571428656578064
acc : 0.8571428571428571
f1 : 0.5970961887477314

SMOTE
loss : 0.44192057847976685
accuracy : 0.8857142925262451
acc : 0.8857142857142857
f1 : 0.6259259259259259

'''
