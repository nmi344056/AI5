# https://www.kaggle.com/competitions/bike-sharing-demand/data?select=train.csv

import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

#1. 데이터
path = "C:\\ai5\\_data\\kaggle\\bike-sharing-demand\\"
# path = "C://ai5//_data//kaggle//bike-sharing-demand//"
# path = "C:/ai5/_data/kaggle/bike-sharing-demand/"

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

print(train_csv.info())             # 결측치가 없다 (Non-Null Count가 모두 10886 non-null)
print(test_csv.info())              # 결측치가 없다 (Non-Null Count가 모두 6493 non-null)
print(train_csv.describe())
'''
             season       holiday    workingday  ...        casual    registered         count
count  10886.000000  10886.000000  10886.000000  ...  10886.000000  10886.000000  10886.000000
mean       2.506614      0.028569      0.680875  ...     36.021955    155.552177    191.574132
std        1.116174      0.166599      0.466159  ...     49.960477    151.039033    181.144454
min        1.000000      0.000000      0.000000  ...      0.000000      0.000000      1.000000
25%        2.000000      0.000000      0.000000  ...      4.000000     36.000000     42.000000
50%        3.000000      0.000000      1.000000  ...     17.000000    118.000000    145.000000
75%        4.000000      0.000000      1.000000  ...     49.000000    222.000000    284.000000
max        4.000000      1.000000      1.000000  ...    367.000000    886.000000    977.000000
# '''

########## 결측치 확인 ##########
print(train_csv.isna().sum())       # 모두 0, 결측치가 없다
print(train_csv.isnull().sum())     # 모두 0, 결측치가 없다
print(test_csv.isna().sum())        # 모두 0, 결측치가 없다
print(test_csv.isnull().sum())      # 모두 0, 결측치가 없다

########## x와 y를 분리 ##########
x = train_csv.drop(['casual', 'registered', 'count'], axis=1)
print(x)                            # [10886 rows x 8 columns]
print(x.shape)                      # (10886, 8)
y = train_csv['count']
print(y)
print(y.shape)                      # (10886,)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.85, random_state=111)

#2. 모델구성
# model = Sequential()
# model.add(Dense(64, activation='relu', input_dim=8))
# model.add(Dense(32, activation='relu'))
# model.add(Dense(16, activation='relu'))
# model.add(Dense(8, activation='relu'))
# model.add(Dense(4, activation='relu'))
# model.add(Dense(1, activation='linear'))

model = Sequential()
model.add(Dense(64, input_dim=8))
model.add(Dense(32))
model.add(Dense(16))
model.add(Dense(8))
model.add(Dense(4))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=50)

#4. 평가, 예측
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
