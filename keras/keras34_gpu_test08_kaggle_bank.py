import tensorflow as tf
print(tf.__version__)       # 2.7.4

gpus = tf.config.experimental.list_physical_devices('GPU')
print(gpus)
# [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
# keras21_2_kaggle_bank copy

# https://www.kaggle.com/competitions/playground-series-s4e1/data?select=train.csv

import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input
from sklearn.model_selection import train_test_split
import time
from sklearn.metrics import r2_score, accuracy_score
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# #0 replace data
# PATH = 'C:/AI5/_data/kaglle/playground-series-s4e1/'

# train_csv = pd.read_csv(PATH + "train.csv", index_col = 0)

# print(train_csv['Geography'].value_counts())

# train_csv['Geography'] = train_csv['Geography'].replace('France', value = 1)
# train_csv['Geography'] = train_csv['Geography'].replace('Spain', value = 2)
# train_csv['Geography'] = train_csv['Geography'].replace('Germany', value = 3)

# train_csv['Gender'] = train_csv['Gender'].replace('Male', value = 1)
# train_csv['Gender'] = train_csv['Gender'].replace('Female', value = 2)

# train_csv.to_csv(PATH + "replaced_train.csv")

# #1 replace data
# PATH = 'C:/AI5/_data/kaglle/playground-series-s4e1/'

# test_csv = pd.read_csv(PATH + "test.csv", index_col = 0)

# print(test_csv['Geography'].value_counts())

# test_csv['Geography'] = test_csv['Geography'].replace('France', value = 1)
# test_csv['Geography'] = test_csv['Geography'].replace('Spain', value = 2)
# test_csv['Geography'] = test_csv['Geography'].replace('Germany', value = 3)

# test_csv['Gender'] = test_csv['Gender'].replace('Male', value = 1)
# test_csv['Gender'] = test_csv['Gender'].replace('Female', value = 2)

# test_csv.to_csv(PATH + "replaced_test.csv")



#1. 데이터
path = 'C:/AI5/_data/kaglle/playground-series-s4e1/'

train_csv = pd.read_csv(path + 'replaced_train.csv', index_col=0)
print(train_csv)    # [165034 rows x 13 columns]

test_csv = pd.read_csv(path + 'replaced_test.csv', index_col=0)
print(test_csv)     # [110023 rows x 12 columns]

submission_csv = pd.read_csv(path + "sample_submission.csv", index_col=0)
print(submission_csv)      # [110023 rows x 1 columns]

print(train_csv.shape)      # (165034, 13)
print(test_csv.shape)       # (110023, 12)
print(submission_csv.shape) # (110023, 1)

print(train_csv.columns)

train_csv.info()    # 결측치 없음
test_csv.info()     # 결측치 없음

test_csv = test_csv.drop(['CustomerId', 'Surname'], axis=1)

###############################################
train_csv=train_csv.drop(['CustomerId', 'Surname'], axis=1)

from sklearn.preprocessing import MinMaxScaler

train_scaler = MinMaxScaler()

train_csv_copy = train_csv.copy()

train_csv_copy = train_csv_copy.drop(['Exited'], axis = 1)

train_scaler.fit(train_csv_copy)

train_csv_scaled = train_scaler.transform(train_csv_copy)

train_csv = pd.concat([pd.DataFrame(data = train_csv_scaled), train_csv['Exited']], axis = 1)

test_scaler = MinMaxScaler()

test_csv_copy = test_csv.copy()

test_scaler.fit(test_csv_copy)

test_csv_scaled = test_scaler.transform(test_csv_copy)

test_csv = pd.DataFrame(data = test_csv_scaled)
###############################################

x = train_csv.drop(['Exited'], axis = 1)
print(x)    # [165034 rows x 10 columns]

y = train_csv['Exited']
print(y.shape)      # (165034,)


print(np.unique(y, return_counts=True))
print(type(x))      # (array([0, 1], dtype=int64), array([424, 228], dtype=int64))

print(pd.DataFrame(y).value_counts())
# 0      424
# 1      228
pd.value_counts(y)

x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.8,
                                                    random_state=1186)

print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)


from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
# scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
scaler = RobustScaler()

scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
test_csv = scaler.transform(test_csv)

print('x_train :', x_train)
print(np.min(x_train), np.max(x_train))
print(np.min(x_test), np.max(x_test))

# #2. 모델구성
# model = Sequential()
# model.add(Dense(16, activation='relu', input_dim=10))
# model.add(Dense(32, activation='relu'))
# model.add(Dense(64, activation='relu'))
# model.add(Dropout(0.3))
# model.add(Dense(128, activation='relu'))
# model.add(Dropout(0.3))
# model.add(Dense(128, activation='relu'))
# model.add(Dropout(0.3))
# model.add(Dense(128, activation='relu'))
# model.add(Dropout(0.3))
# model.add(Dense(64, activation='relu'))
# model.add(Dropout(0.3))
# model.add(Dense(32, activation='relu'))
# model.add(Dense(16, activation='relu'))
# model.add(Dense(8, activation='relu'))
# model.add(Dense(4, activation='relu'))
# model.add(Dense(2, activation='relu'))

# model.add(Dense(1, activation='sigmoid'))

#2-2. 모델구성(함수형)
input1 = Input(shape=(10, ))
dense1 = Dense(16,activation='relu',name='ys1')(input1)
dense2 = Dense(32, activation='relu',name='ys2')(dense1)
dense3 = Dense(64, activation='relu', name='ys3')(dense2)
drop1 = Dropout(0.3)(dense3)
dense4 = Dense(128, activation='relu', name='ys4')(drop1)
drop2 = Dropout(0.3)(dense4)
dense5 = Dense(128, activation='relu', name='ys5')(drop2)
drop3 = Dropout(0.3)(dense5)
dense6 = Dense(128, activation='relu', name='ys6')(drop3)
drop4 = Dropout(0.3)(dense6)
dense7 = Dense(64, activation='relu', name='ys7')(drop4)
drop5 = Dropout(0.3)(dense7)
dense10 = Dense(32, activation='relu', name='ys10')(drop5)
dense11= Dense(16, activation='relu', name='ys11')(dense10)
dense12 = Dense(8, activation='relu', name='ys12')(dense11)
dense13= Dense(4, activation='relu', name='ys13')(dense12)
dense14= Dense(2, activation='relu', name='ys14')(dense13)
output1 = Dense(1, activation='sigmoid', name='ys15')(dense14)
model=Model(inputs=input1, outputs=output1)
model.summary()





#3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
start = time.time()

es = EarlyStopping(
    monitor = 'val_loss',
    mode = 'min',
    patience = 40,
    restore_best_weights=True
)


import datetime
date = datetime.datetime.now()       # 현재시간 저장
print(date)         # 2024-07-26 16:50:37.570567
print(type(date))   # <class 'datetime.datetime'>
date = date.strftime("%m%d_%H%M")
print(date)
print(type(date)) # <class 'str'>


path ='C:/AI5/_save/keras32_mcp/08_kaggle_bank/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'    # 1000-0.7777.hdf5    { } = dictionary, 키와 밸류  d는 정수, f는 소수
filepath = "".join([path, 'k32_08', date, '_', filename])      # 파일위치와 이름을 에포와 발로스로 나타내준다
# 생성 예 : ""./_save/keras29_mcp/k29_1000-0.7777.hdf5"
##################### MCP 세이브 파일명 만들기 끝 ###########################

mcp = ModelCheckpoint(
    monitor='val_loss',
    mode='auto',
    verbose = 1,
    save_best_only=True,
    filepath = filepath
)

hist = model.fit(x_train, y_train, epochs=1000, batch_size=555,
                 validation_split=0.2,
                 callbacks=[es, mcp]
                 )
end = time.time()

#4. 평가, 예측
loss = model.evaluate(x_test, y_test, verbose=1)
print('로스 :', loss)
print("acc :", round(loss[1],3))

y_pred = model.predict(x_test)
print(y_pred[:50])
y_pred = np.round(y_pred)
print(y_pred[:50])

acc_score = accuracy_score(y_test, y_pred)

print('acc_score :', acc_score)
print('걸린시간 : ', round(end - start, 2), "초")

print(test_csv.shape)


if(gpus):
    print("쥐피유 돈다!!!")
else:
    print("쥐피유 없다! xxxxx")

print('걸린시간 : ', round(end - start, 2), "초")

# 쥐피유 없다! xxxxx
# 걸린시간 :  49.26 초

# 쥐피유 돈다!!!
# 걸린시간 :  175.01 초