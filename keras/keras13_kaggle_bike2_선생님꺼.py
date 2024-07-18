# https://www.kaggle.com/competitions/bike-sharing-demand/data?select=train.csv

import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

#1. 데이터
path = 'C:\\AI5\\_data\\bike-sharing-demand\\'      # 절대경로
# path = 'C://AI5//_data//bike-sharing-demand//'      # 절대경로   다 가능
# path = 'C:/AI5/_data/bike-sharing-demand/'      # 절대경로
#  /  //  \  \\ 다 가능

train_csv = pd.read_csv(path + "train.csv", index_col=0)
test_csv = pd.read_csv(path + "test.csv", index_col=0)
sampleSubmission = pd.read_csv(path + "sampleSubmission.csv", index_col=0)

# train_dt = pd.DatetimeIndex(train_csv.index)

# train_csv['day'] = train_dt.day
# train_csv['month'] = train_dt.month
# train_csv['year'] = train_dt.year
# train_csv['hour'] = train_dt.hour
# train_csv['dow'] = train_dt.dayofweek

# test_dt = pd.DatetimeIndex(test_csv.index)

# test_csv['day'] = test_dt.day
# test_csv['month'] = test_dt.month
# test_csv['year'] = test_dt.year
# test_csv['hour'] = test_dt.hour
# test_csv['dow'] = test_dt.dayofweek


print(train_csv.shape)      # (10886, 11)
print(test_csv.shape)       # (6493, 8)
print(sampleSubmission.shape)   # (6493, 1)
# datetime,season,holiday,workingday,weather,temp,atemp,humidity,windspeed
# datetime,season,holiday,workingday,weather,temp,atemp,humidity,windspeed,casual,registered,count

print(train_csv.columns)       # 'season', 'holiday', 'workingday', 'weather', 'temp', 'atemp',
    #    'humidity', 'windspeed', 'casual', 'registered', 'count'

print(train_csv.info())
print(test_csv.info())

print(train_csv.describe().T)   # describe 평균,중위값 등등 나타냄. 많이쓴다.

############### 결측치 확인 #################
print(train_csv.isna().sum())
print(train_csv.isnull().sum())
print(test_csv.isna().sum())             # 전부 결측치 없음 확인
print(test_csv.isnull().sum())

############# x와 y를 분리 ########
x = train_csv.drop(['casual', 'registered','count'], axis=1)    # 대괄호 하나 = 리스트    두개 이상은 리스트
print(x)            # [10886 rows x 8 columns]

y = train_csv[['casual', 'registered']]
print(y.shape)      #(10886, 2)

x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.95,
                                                    random_state = 7979)

print('x_train :', x_train.shape)       #(10341, 8)
print('x_test :', x_test.shape)         #(545, 8)
print('y_train :', y_train.shape)       #(10341, 2)
print('y_test :', y_test.shape)         #(545, 2)


#2. 모델구성
model = Sequential()
model.add(Dense(10, activation='relu', input_dim=8))
model.add(Dense(20, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(40, activation='relu'))
model.add(Dense(60, activation='relu'))
model.add(Dense(80, activation='relu'))
model.add(Dense(80, activation='relu'))
model.add(Dense(80, activation='relu'))
model.add(Dense(80, activation='relu'))
model.add(Dense(40, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(10, activation='relu'))

model.add(Dense(2, activation='linear'))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs = 1000, batch_size=333)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('로스 :', loss)

y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print("r2 스코어 :", r2)

print(test_csv.shape)   # (6493, 8)
y_submit = model.predict(test_csv)
print(y_submit)
print(y_submit.shape)   # (6493, 2)

print("test_csv타입 :", type(test_csv))       # test_csv타입 : <class 'pandas.core.frame.DataFrame'>
print("y_submit타입:", type(y_submit))        # y_submit타입: <class 'numpy.ndarray'>

test2_csv = test_csv
print(test2_csv.shape)      # (6493, 8)

test2_csv[['casual', 'registered']] = y_submit
print(test2_csv)    # [6493 rows x 10 columns]

test2_csv.to_csv(path + "test2.csv")





# sampleSubmission['count'] = y_submit
# print(sampleSubmission)         # [6493 rows x 1 columns]
# print(sampleSubmission.shape)   # (6493, 1)

# sampleSubmission.to_csv(path + "sampleSubmission_0718_1202.csv")

# print('로스 :', loss)
# print("r2 스코어 :", r2)
# '''
