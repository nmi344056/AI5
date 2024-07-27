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
mission_csv = pd.read_csv(path + "sampleSubmission.csv", index_col=0)

# print(train_csv.shape)      # (10886, 11)
# print(test_csv.shape)       # (6493, 8)
# print(mission_csv.shape)        # (6493, 1)

print(train_csv.columns)
# Index(['season', 'holiday', 'workingday', 'weather', 'temp', 'atemp',
#        'humidity', 'windspeed', 'casual', 'registered', 'count'],
#       dtype='object')

########## 결측치, 이상치 확인 ##########
print(train_csv.info())         # 결측치가 없다.
print(test_csv.info())          # 결측치가 없다.
print(train_csv.isna().sum())   # 0
print(test_csv.isna().sum())    # 0
print(train_csv.describe())     # 이상치 존재 여부 확인

########## x와 y를 분리 ##########
x = train_csv.drop(['casual', 'registered', 'count'], axis=1)
print(x.shape)    # (10886, 8)
y = train_csv[['casual', 'registered']]     # 2차원 형태
print(y)
print(y.shape)    # (10886, 2)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=123)
print(x_train.shape, x_test.shape)          # (8708, 8) (2178, 8)
print(y_train.shape, y_test.shape)          # (8708, 2) (2178, 2)

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
model.fit(x_train, y_train, epochs=1000, batch_size=30)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print("loss : ", loss)

y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print("re score : ", r2)

print(test_csv.shape)           # (6493, 8)
y_submit = model.predict(test_csv)
print(y_submit)
print(y_submit.shape)           # (6493, 2)

print("test_csv type : ", type(test_csv))   # <class 'pandas.core.frame.DataFrame'>
print("y_submit type : ", type(y_submit))   # <class 'numpy.ndarray'>

print(test_csv)
'''
                     season  holiday  workingday  weather   temp   atemp  humidity  windspeed        
datetime
2011-01-20 00:00:00       1        0           1        1  10.66  11.365        56    26.0027        
2011-01-20 01:00:00       1        0           1        1  10.66  13.635        56     0.0000        
2011-01-20 02:00:00       1        0           1        1  10.66  13.635        56     0.0000             
...                     ...      ...         ...      ...    ...     ...       ...        ...              
2012-12-31 21:00:00       1        0           1        1  10.66  12.880        60    11.0014        
2012-12-31 22:00:00       1        0           1        1  10.66  13.635        56     8.9981        
2012-12-31 23:00:00       1        0           1        1  10.66  13.635        65     8.9981 
'''
print(y_submit)
'''
[[44.845505 70.2779  ]
 [21.967867 34.693867]
 [21.967867 34.693867]
 ...
 [28.86848  45.427128]
 [29.794094 46.866833]
 [26.563503 41.84195 ]]
'''

test2_csv = test_csv
print(test2_csv.shape)          # (6493, 8)

test2_csv[['casual', 'registered']] = y_submit
print(test2_csv)
print(test2_csv.shape)          # (6493, 10)

test2_csv.to_csv(path + "test2.csv")
