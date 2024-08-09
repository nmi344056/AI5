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
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM, GRU
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

#1. 데이터
path = "C:\\ai5\\_data\\kaggle\\jena\\"
csv = pd.read_csv(path + "jena_climate_2009_2016.csv", index_col=0)
submit = pd.read_csv(path + "jena_climate_2009_2016.csv")

print(csv.shape)            # (420551, 14)

y2 = csv.tail(144)
y2 = y2['T (degC)']

csv = csv[:-144]
# print(csv)
# print(csv.shape)          # (420407, 14)

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
# print(x.shape)            # (419688, 720, 13)

y = split_x(y, size)
# print(y)
# print(y.shape)            # (420264, 144)

# x_predict = x[-1:]        # (1, 144, 13)
x_predict = x[-1]           # (144, 13)
x_predict = np.array(x_predict).reshape(1, 144, 13)
# print(x_predict.shape)    # (1, 144, 13)

x = x[:-144, :]
y = y[144:]
# print(x.shape)            # (420263, 144, 13)
# print(y.shape)            # (420263, 144)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=123)
# print(x_train.shape, x_test.shape)  # (336210, 144, 13) (84053, 144, 13)
# print(y_train.shape, y_test.shape)  # (336210, 144) (84053, 144)

# # #2. 모델구성
# model = Sequential()
# model.add(LSTM(16, input_shape=(144,13), return_sequences=True))
# model.add(LSTM(32))
# model.add(Dense(64))
# model.add(Dense(128))
# model.add(Dense(64))
# model.add(Dense(32))
# model.add(Dense(16))
# model.add(Dense(144))

# #3. 컴파일, 훈련
# model.compile(loss='mse', optimizer='adam')

# es = EarlyStopping(monitor='val_loss', mode='min', patience=20, verbose=1, restore_best_weights=True,)

# mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1, save_best_only=True, filepath=filepath)

# start = time.time()
# model.fit(x_train, y_train, epochs=1000, batch_size=1024, validation_split=0.2, callbacks=[es, mcp])
# end = time.time()

#4. 평가, 예측
path_w = 'C:\\ai5\\_save\\keras58\\'
# model = load_model(path_w + 'jena_안혜지.hdf5')
model = load_model(path_w + 'k55_jena_date_0812.1447_epo_0040_valloss_0.4722.hdf5')

model.summary()

loss = model.evaluate(x_test, y_test)
print("loss : ", loss)

y_predict = model.predict(x_predict)
# print(y_predict.shape)  # (1, 144)

# y_predict = np.array(y_predict).reshape(144, 1)
y_predict = y_predict.T
# print(y_predict.shape)  # (144, 1)

def RMSE(y_test,y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
rmse = RMSE(y2, y_predict)
print('RMSE : ', rmse)          # 결과 : RMSE :  1.163784915583028

submit = submit[['Date Time','T (degC)']]
submit = submit.tail(144)
# print(submit)

# y_submit = pd.DataFrame(y_predict)
# print(y_submit)

submit['T (degC)'] = y_predict
# print(submit)                  # [6493 rows x 1 columns]
# print(submit.shape)            # (6493, 1)

submit.to_csv(path_w + "jena_안혜지.csv", index=False)

'''
제출
loss :  0.1985899657011032
RMSE :  1.163784915583028
'''
