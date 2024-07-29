# https://www.kaggle.com/competitions/bike-sharing-demand/data?select=train.csv

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from sklearn.metrics import r2_score, accuracy_score
import numpy as np
import pandas as pd
import time

#1. 데이터
path = "C:\\ai5\\_data\\kaggle\\bike-sharing-demand\\"
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

scaler = StandardScaler()

x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
test_csv = scaler.transform(test_csv)

print(x_train)
print(np.min(x_train), np.max(x_train))     # 0.0 1.0
print(np.min(x_test), np.max(x_test))       # 0.0 1.017298281552293

#2. 모델구성
input1 = Input(shape=(8,))
dense1 = Dense(128)(input1)
drop1 = Dropout(0.3)(dense1)
dense2 = Dense(64, activation='relu')(drop1)
drop2 = Dropout(0.3)(dense2)
dense3 = Dense(32, activation='relu')(drop2)
drop3 = Dropout(0.2)(dense3)
dense4 = Dense(32, activation='relu')(drop3)
drop4 = Dropout(0.2)(dense4)
dense5 = Dense(32, activation='relu')(drop4)
output1 = Dense(1, activation='linear')(dense5)
model = Model(inputs=input1, outputs=output1)

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

es = EarlyStopping(monitor='val_loss', mode='min', patience=50, verbose=1, restore_best_weights=True,)

########## mcp 세이브 파일명 만들기 시작 ##########
import datetime
date = datetime.datetime.now()
print(date)
print(type(date))

date = date.strftime("%m%d.%H%M")
print(date)
print(type(date))

path = './_save/keras32/'
filename = '{epoch:04d}_valloss_{val_loss:.4f}.hdf5'
filepath = "".join([path, 'k32_05_date_', date, '_epo_', filename])

########## mcp 세이브 파일명 만들기 끝 ##########

mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1, save_best_only=True, filepath=filepath)

start = time.time()
hist = model.fit(x_train, y_train, epochs=500, batch_size=100, validation_split=0.2, callbacks=[es, mcp])
end = time.time()

#4. 평가, 예측
print("++++++++++++++++++++")
loss = model.evaluate(x_test, y_test)
print("loss : ", loss)

y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print("re score : ", r2)
print("time : ", round(end - start, 2), "초")

y_submit = model.predict(test_csv)
print(y_submit)
print(y_submit.shape)                 # (6493, 1)

########## sampleSubmission.csv 만들기 (count컬럼에 값만 넣으면 된다) ##########
sampleSubmission['count'] = y_submit
print(sampleSubmission)               # [6493 rows x 1 columns]
print(sampleSubmission.shape)         # (6493, 1)

sampleSubmission.to_csv(path + "sampleSubmission_0729_13.csv")

'''
128 64 32 32 32 1 / train_size=0.85, random_state=111 / epochs=500, batch_size=100

loss :  21271.451171875
re score :  0.35337467674572753

CPU
loss :  22430.201171875
re score :  0.3181501394248807
time :  12.37 초

GPU
loss :  21895.673828125
re score :  0.3343992261488593
time :  37.53 초
'''
