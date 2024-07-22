# https://www.kaggle.com/competitions/bike-sharing-demand/data?select=train.csv

import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import time
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

y = train_csv['count']
print(y.shape)      #(10886,)

x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.9,
                                                    random_state = 68481)

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
model.add(Dense(80, activation='relu'))
model.add(Dense(80, activation='relu'))
model.add(Dense(80, activation='relu'))
model.add(Dense(80, activation='relu'))
model.add(Dense(80, activation='relu'))
model.add(Dense(80, activation='relu'))
model.add(Dense(80, activation='relu'))
model.add(Dense(80, activation='relu'))
model.add(Dense(80, activation='relu'))
model.add(Dense(80, activation='relu'))
model.add(Dense(80, activation='relu'))
model.add(Dense(80, activation='relu'))
model.add(Dense(80, activation='relu'))
model.add(Dense(80, activation='relu'))
model.add(Dense(80, activation='relu'))
model.add(Dense(80, activation='relu'))
model.add(Dense(80, activation='relu'))
model.add(Dense(80, activation='relu'))
model.add(Dense(80, activation='relu'))
model.add(Dense(80, activation='relu'))
model.add(Dense(80, activation='relu'))
model.add(Dense(80, activation='relu'))
model.add(Dense(80, activation='relu'))
model.add(Dense(80, activation='relu'))
model.add(Dense(40, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(10, activation='relu'))

model.add(Dense(1, activation='linear'))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
start = time.time()

from tensorflow.keras.callbacks import EarlyStopping 
es = EarlyStopping(                                               
    monitor = 'val_loss',                                              
    mode = 'min',
    patience = 20,
    restore_best_weights = True
)
hist = model.fit(x_train, y_train, epochs = 600, batch_size=50,
                 validation_split=0.25,
                 callbacks=[es])
end = time.time()

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('로스 :', loss)

y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print("r2 스코어 :", r2)
print("걸린시간 :", round(end-start,2),"초")

y_submit = model.predict(test_csv)
print(y_submit)
print(y_submit.shape)

# sampleSubmission['count'] = y_submit
# print(sampleSubmission)         # [6493 rows x 1 columns]
# print(sampleSubmission.shape)   # (6493, 1)

# sampleSubmission.to_csv(path + "sampleSubmission_0718_1202.csv")

# print('로스 :', loss)
# print("r2 스코어 :", r2)

print("============================= hist =========================")
print(hist)
print("============================= hist.history ==================")
print(hist.history)
print("============================= loss ==================")
print(hist.history['loss'])
print("============================= val_loss ==================")
print(hist.history['val_loss'])
print("==========================================================")

import matplotlib.pyplot as plt
plt.rcParams['font.family'] ='Malgun Gothic'    # 한글깨짐 해결
plt.figure(figsize=(9,6))       # 그림판 사이즈
plt.plot(hist.history['loss'], c='red', label='loss')
plt.plot(hist.history['val_loss'], c='blue', label='val_loss')
plt.legend(loc='upper right')
plt.title('바이크 loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.grid()      # 모눈종이
plt.show()
