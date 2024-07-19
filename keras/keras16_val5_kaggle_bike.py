# https://www.kaggle.com/competitions/bike-sharing-demand/data?select=train.csv

import numpy as np
import pandas as pd

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

#1. 데이터
path = "C:\\ai5\\_data\\bike-sharing-demand\\"
# path = "C://ai5//_data//bike-sharing-demand//"
# path = "C://ai5/_data/bike-sharing-demand/"

train_csv = pd.read_csv(path + "train.csv", index_col=0)
test_csv = pd.read_csv(path + "test.csv", index_col=0)
sampleSubmission = pd.read_csv(path + "sampleSubmission.csv", index_col=0)

print(train_csv.shape)              # (10886, 11)
print(test_csv.shape)               # (6493, 8)
print(sampleSubmission.shape)       # (6493, 1)

print(train_csv.columns)
# Index(['season', 'holiday', 'workingday', 'weather', 'temp', 'atemp',
#        'humidity', 'windspeed', 'casual', 'registered', 'count'],    
#       dtype='object')

print(train_csv.info())             # 결측치가 없다
print(test_csv.info())              # 결측치가 없다
print(train_csv.describe())

########## 결측치 확인 ##########
print(train_csv.isna().sum())       # 0
print(train_csv.isnull().sum())     # 0
print(test_csv.isna().sum())        # 0
print(test_csv.isnull().sum())      # 0

########## x와 y를 분리 ##########
x = train_csv.drop(['casual', 'registered', 'count'], axis=1)
print(x)                            # [10886 rows x 8 columns]
y = train_csv['count']
print(y)
print(y.shape)                      # (10886,)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.85, random_state=111)

#2. 모델구성
model = Sequential()
model.add(Dense(64, activation='relu', input_dim=8))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(1, activation='linear'))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=50, validation_split=0.2)

#4. 평가, 예측
print("++++++++++++++++++++")
loss = model.evaluate(x_test, y_test)
print("loss : ", loss)

y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print("re score : ", r2)

y_submit = model.predict(test_csv)
print(y_submit)
print(y_submit.shape)                 # (6493, 1)

########## sampleSubmission.csv 만들기 (count컬럼에 값만 넣으면 된다) ##########
sampleSubmission['count'] = y_submit
print(sampleSubmission)               # [6493 rows x 1 columns]
print(sampleSubmission.shape)         # (6493, 1)

sampleSubmission.to_csv(path + "sampleSubmission_0717_1.csv")

print("loss : ", loss)

'''
64 32 32 16 1 / train_size=0.8, random_state=123 / epochs=100, batch_size=1000 / loss :  23073.974609375 (제출1)    1.27
100 50 25 10 1 / train_size=0.8, random_state=434 / epochs=100, batch_size=100 / loss :  21900.65625 (제출2)        1.32
'''
