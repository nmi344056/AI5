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

np.random.seed(337)         # numpy seed 고정
import tensorflow as tf
tf.random.set_seed(337)     # tensorflow seed 고정
import random as rn
rn.seed(337)                # python seed 고정

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

lr = [0.1, 0.01, 0.005, 0.001, 0.0005, 0.0001]
results = []

from tensorflow.keras.optimizers import Adam
for i in range(len(lr)):

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

    #3. 컴파일, 훈련
    model.compile(loss='mse', optimizer=Adam(learning_rate=lr[i]), metrics=['accuracy'])

    es = EarlyStopping(monitor='val_loss', mode='min', patience=20, verbose=1, restore_best_weights=True,)

    ########## mcp 세이브 파일명 만들기 시작 ##########
    import datetime
    date = datetime.datetime.now()
    date = date.strftime("%m%d_%H%M")

    path = './_save/keras68/'
    filename = '{epoch:04d}_valloss_{val_loss:.4f}.hdf5'
    filepath = "".join([path, 'k68_10_date_', str(i+1), '_', date, '_epo_', filename])

    ########## mcp 세이브 파일명 만들기 끝 ##########

    mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1, save_best_only=True, filepath=filepath)

    start = time.time()
    model.fit(x_train, y_train, epochs=1, batch_size=1024, validation_split=0.2, verbose=2, callbacks=[es, mcp])
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
2번째부터 에러
tensorflow.python.framework.errors_impl.InternalError: Failed copying input tensor from /job:localhost/replica:0/task:0/device:CPU:0 to /job:localhost/replica:0/task:0/device:GPU:0 in order to run _EagerConst: Dst tensor is not initialized.


'''
