# [실습] jena을 DNN으로 구성
# x : (42만, 144,13) -> (42만, 144*13)    x를 2차원으로
# y : (42만, 144)

# https://www.kaggle.com/datasets/stytch16/jena-climate-2009-2016/code
# https://www.kaggle.com/code/tila123/starter-jena-climate-2009-2016-627dd05e-6/notebook

'''
[실습] y는 T (degC), 자를는거는 마음대로
31.12.2016 00:10:00 부터 01.01.2017 00:00:00 까지는 사용 X
x shape = (n, 720(5일*144), 13), y shape = (n, 144), predict = (1, 144)
평가지표 : RMSE
'''

import numpy as np
import pandas as pd
import time
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM, GRU
from tensorflow.keras.layers import Conv1D, Flatten, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

#1. 데이터
path = "C:\\ai5\\_data\\kaggle\\jena\\"
csv = pd.read_csv(path + "jena_climate_2009_2016.csv", index_col=0)
print(csv.shape)            # (420551, 14)

y2 = csv.tail(144)
y2 = y2['T (degC)']

csv = csv[:-144]
# print(csv)
print(csv.shape)          # (420407, 14)

x = csv.drop(['T (degC)'], axis=1)
y = csv['T (degC)']
print(x.shape, y.shape)   # (420407, 13) (420407,)

size = 144

def split_x(dataset, size):
    aaa = []
    for i in range(len(dataset) - size + 1):
        subset = dataset[i : (i + size)]
        aaa.append(subset)
    return np.array(aaa)

x = split_x(x, size)
# print(x)
print(x.shape)            # (420264, 144, 13)

y = split_x(y, size)
# print(y)
print(y.shape)            # (420264, 144)

# x_predict = x[-1:]        # (1, 144, 13)
x_predict = x[-1]           # (144, 13)
x_predict = np.array(x_predict).reshape(1, 144, 13)
# print(x_predict.shape)    # (1, 144, 13)

x = x[:-1, :]
y = y[1:]
# print(x.shape)            # (420263, 144, 13)
# print(y.shape)            # (420263, 144)

x = x.reshape(420263, 144*13)
x_predict = x_predict.reshape(1, 144*13)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=123)
# print(x_train.shape, x_test.shape)  # (336210, 144, 13) (84053, 144, 13)
# print(y_train.shape, y_test.shape)  # (336210, 144) (84053, 144)

#2. 모델구성
model = Sequential()
model.add(Dense(16, activation='relu', input_shape=(144*13,)))
model.add(Dense(32, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(144))

# model = Sequential()
# model.add(LSTM(32, return_sequences=True, input_shape=(144, 13)))
# model.add(LSTM(120))
# model.add(Dense(400, activation='relu'))
# model.add(Dropout(0.1))
# model.add(Dense(370, activation='relu'))
# model.add(Dropout(0.1)) 
# model.add(Dense(320, activation='relu'))
# model.add(Dense(300, activation='relu'))
# model.add(Dense(288, activation='relu'))
# model.add(Dense(144))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

es = EarlyStopping(monitor='val_loss', mode='min', patience=20, verbose=1, restore_best_weights=True,)

########## mcp 세이브 파일명 만들기 시작 ##########
import datetime
date = datetime.datetime.now()
print(date)
print(type(date))

date = date.strftime("%m%d.%H%M")
print(date)
print(type(date))

path = './_save/keras61/'
filename = '{epoch:04d}_valloss_{val_loss:.4f}.hdf5'
filepath = "".join([path, 'k61_jena_date_', date, '_epo_', filename])

########## mcp 세이브 파일명 만들기 끝 ##########

mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1, save_best_only=True, filepath=filepath)

start = time.time()
model.fit(x_train, y_train, epochs=1000, batch_size=1024, validation_split=0.2, verbose=2, callbacks=[es, mcp])
end = time.time()

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print("loss :", loss)

y_predict = model.predict(x_predict)
# print(y_predict.shape)  # (1, 144)

# y_predict = np.array(y_predict).reshape(144, 1)
y_predict = y_predict.T
# print(y_predict.shape)  # (144, 1)

def RMSE(y_test,y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
rmse = RMSE(y2, y_predict)
print('RMSE :', rmse)

print("time :", round(end-start,2),'초')
print('loss :',loss, '/ RMSE :',rmse, '(',round(end-start,2),'초)')

# submit_csv['T (degC)'] = y_submit
# print(submit_csv)                  # [6493 rows x 1 columns]
# print(submit_csv.shape)            # (6493, 1)

# submit_csv.to_csv(path + "sampleSubmission_0809.csv")

'''
16 32 64 128 64 32 16, batch=1024, p=20 / time : 1010.2 초 ~ 1768.91 초

loss : 0.1985899657011032 / RMSE : 1.163784915583028 (1608.57 초)
k55_jena_date_0810.1827_epo_0265_valloss_0.1945     * (submit)

loss : 0.1594621241092682 / RMSE : 1.223160881069847 (1768.91 초)
k55_jena_date_0812.1226_epo_0309_valloss_0.1625

-----
32 120 400 Drop 370 Drop 320 300 288

loss : 0.48345789313316345 / RMSE : 1.1140172949585425 ( 485.38 초)
k55_jena_date_0812.1447_epo_0040_valloss_0.4722

-----
dropout을 사용
처음에는 RMSE가 1.4~1.7로 높게나왔는데 반복해서 돌리니까 1.11~1.2로 낮아져서 RMSE 갱신.
또한 시간이 1/2~1/3로 단축되어 동일한 시간안에 여러번 훈련이 가능했다.
(가충치 save 파일의 크기가 16배 증가.)

patience을 늘리거나 batch_size를 줄여봤지만 시간만 길어지고 성능(RMSE)은 좋지 않았다.
몇가지 model을 돌려본 결과 위 2개 model의 성능이 가장 좋았다.

-----
Conv1D
loss : 0.28821897506713867 / RMSE : 1.2338752904367398 ( 122.09 초) > k58_jena_date_0813.1112_epo_0067_valloss_0.2802
loss : 0.18998293578624725 / RMSE : 1.2625251699921636 ( 138.62 초) > k58_jena_date_0813.1124_epo_0081_valloss_0.1855
loss : 0.22925937175750732 / RMSE : 1.1852023987578018 ( 129.49 초) > k58_jena_date_0813.1144_epo_0074_valloss_0.2263

DNN
loss : 0.5098708271980286 / RMSE : 1.5352259135739297 ( 114.83 초) > k58_jena_date_0814.1032_epo_0079_valloss_0.5043
(속도는 훨씬 빠르다)

'''
