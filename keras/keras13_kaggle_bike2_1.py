'''
기존 kaggle 데이터에서
1. train_csv의 y를 casual과 register로 잡는다.
    그래서 훈련을 해서 test_csv의 casual과 register를 predict한다.
2. test_csv에 casual과 register 컬럼을 합쳐
3. train_csv에 y를 count로 잡는다.
4. 전체 훈련
5. test_csv 예측해서 submission에 붙인다
'''

import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

#1. 데이터
path = "C:\\ai5\\_data\\kaggle\\bike-sharing-demand\\"
train_csv = pd.read_csv(path + "train.csv", index_col=0)
test_csv = pd.read_csv(path + "test.csv", index_col=0)
mission = pd.read_csv(path + "sampleSubmission.csv", index_col=0)

# print(train_csv.shape)      # (10886, 11)
# print(test_csv.shape)       # (6493, 8)
# print(mission.shape)        # (6493, 1)

print(train_csv.columns)
# Index(['season', 'holiday', 'workingday', 'weather', 'temp', 'atemp',
#        'humidity', 'windspeed', 'casual', 'registered', 'count'],
#       dtype='object')

x = train_csv.drop(['casual', 'registered', 'count'], axis=1)
print(x.shape)    # (10886, 8)
y = train_csv[['casual', 'registered']]     # 2차원 형태
print(y.shape)    # (10886, 2)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=123)

#2. 모델구성
model = Sequential()
model.add(Dense(64, activation='relu', input_dim=8))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(2, activation='linear'))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=1, batch_size=50)

#4. 평가, 예측
print("++++++++++++++++++++")
loss = model.evaluate(x_test, y_test)
print("loss : ", loss)

y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print("re score : ", r2)

y_submit = model.predict(test_csv)
print(y_submit)
print(y_submit.shape)           # (6493, 2)

c_predict = y_submit[:,0]
r_predict = y_submit[:,1]
test_csv = test_csv.assign(casual=c_predict, registered = r_predict)
# test_csv = test_csv.assign(casual=y_submit[:,0], registered = y_submit[:,1])

test_csv.to_csv(path + "test_columnplus.csv")
