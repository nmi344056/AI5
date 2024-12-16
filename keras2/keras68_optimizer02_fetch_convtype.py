from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from sklearn.metrics import r2_score, accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.datasets import fetch_covtype
import numpy as np
import pandas as pd
import time

np.random.seed(337)         # numpy seed 고정
import tensorflow as tf
tf.random.set_seed(337)     # tensorflow seed 고정
import random as rn
rn.seed(337)                # python seed 고정

#1. 데이터
datasets = fetch_covtype()
x = datasets.data
y = datasets.target
print(x.shape, y.shape)     # (581012, 54) (581012,)



print(pd.value_counts(y))    
# 2    283301
# 1    211840
# 3     35754
# 7     20510
# 6     17367
# 5      9493
# 4      2747
# 문제 : 0이 없다, onehot을 0이 아닌 1부터 시작한다.

print(y)
print(np.unique(y, return_counts=True))

# from tensorflow.keras.utils import to_categorical
# y_ohe = to_categorical(y)
# print(y_ohe)
# print(y_ohe.shape)              # (581012, 8)

y_ohe = pd.get_dummies(y)          # pandas
print(y_ohe)                       # 1  2  3  4  5  6  7
print(y_ohe.shape)                 # (581012, 7)

# print("==============================")
# from sklearn.preprocessing import OneHotEncoder
# y_ohe = y.reshape(-1, 1)
# ohe = OneHotEncoder(sparse=False)        # True가 default
# ohe.fit(y_ohe)
# y_ohe = ohe.transform(y_ohe)
# print(y_ohe)
# print(y_ohe.shape)                 # (581012, 7)

x_train, x_test, y_train, y_test = train_test_split(x, y_ohe, train_size=0.9, random_state=6666,
                                                    stratify=y)

# print(pd.value_counts(y_train))
# 2    141429       141651
# 1    106155       105920
# 3     17958       17877
# 7     10262       10255
# 6      8613       8683
# 5      4711       4747
# 4      1378       1373

print(x_train.shape, x_test.shape)      # (522910, 54) (58102, 54)
print(y_train.shape, y_test.shape)      # (522910, 8) (58102, 8)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

lr = [0.1, 0.01, 0.005, 0.001, 0.0005, 0.0001]
results = []

for i in range(len(lr)):

    #2. 모델구성
    model = Sequential()
    model.add(Dense(128, activation='relu', input_shape=(x_train.shape[1],)))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(126, activation='relu'))
    model.add(Dense(7, activation='softmax'))

    #3. 컴파일, 훈련
    from tensorflow.keras.optimizers import Adam
    model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=lr[i]), metrics=['accuracy'])

    es = EarlyStopping(monitor='val_loss', mode='min', patience=10, verbose=1, restore_best_weights=True,)

    ########## mcp 세이브 파일명 만들기 시작 ##########
    import datetime
    date = datetime.datetime.now()
    date = date.strftime("%m%d_%H%M")

    path = './_save/keras68/'
    filename = '{epoch:04d}_valloss_{val_loss:.4f}.hdf5'
    filepath = "".join([path, 'k68_02_date_', str(i+1), '_', date, '_epo_', filename])

    ########## mcp 세이브 파일명 만들기 끝 ##########

    mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1, save_best_only=True, filepath=filepath)

    start = time.time()
    hist = model.fit(x_train, y_train, epochs=500, batch_size=1000, validation_split=0.2, callbacks=[es, mcp])
    end = time.time()

    #4. 평가, 예측
    loss = model.evaluate(x_test, y_test, verbose=1)
    # print("loss : ", loss[0])
    # print("accuracy : ", round(loss[1], 3))

    y_predict = model.predict(x_test)

    r2 = r2_score(y_test, y_predict)

    y_predict = np.round(y_predict)
    acc = accuracy_score(y_test, y_predict)

    print('{0} > loss : {1} / acc : {2} / r2 : {3}'.format(lr[i], loss, acc, r2))

# print(i+1)
# print(lr[i])
# print(round(end-start,2))
# print(loss[0])
# print(loss[1])
# print('==========')


# results.append([i+1, lr[i], round(end-start,2), loss[0], loss[1]])

# for a in results:
#     print('결과', a[0])
#     print('PCA :', a[1])
#     print('time :', a[2],'초')
#     print('loss :', a[3])
#     print('accuracy :', a[4])
#     print('')

# for k in range(0, len(lr), 1):
#     print('결과', results[k][0])
#     print('learning_rate :', results[k][1])
#     print('time :', results[k][2],'초')
#     print('loss :', results[k][3])
#     print('accuracy :', results[k][4])
#     print('')

'''
결과 1
learning_rate : 0.1
time : 1.19 초
loss : 1.0491526126861572
accuracy : 0.4876079857349396

결과 2
learning_rate : 0.01
time : 0.56 초
loss : 0.9455515146255493
accuracy : 0.5988433957099915

결과 3
learning_rate : 0.005
time : 0.64 초
loss : 0.8877478241920471
accuracy : 0.6194279193878174

결과 4
learning_rate : 0.001
time : 0.72 초
loss : 0.8778814673423767
accuracy : 0.6269319653511047

결과 5
learning_rate : 0.0005
time : 0.63 초
loss : 0.8728227019309998
accuracy : 0.6291866302490234

결과 6
learning_rate : 0.0001
time : 0.78 초
loss : 0.871849000453949
accuracy : 0.6275515556335449

'''



'''

0.1 > loss : [0.8787556290626526, 0.5882585644721985] / acc : 0.4880899108464425 / r2 : 0.16887134176864152
0.01 > loss : [0.5527849793434143, 0.7669615745544434] / acc : 0.7409899831331107 / r2 : 0.4173557561187417
0.005 > loss : [0.5314253568649292, 0.7701455950737] / acc : 0.7560324945784999 / r2 : 0.4529745273643075
0.001 > loss : [0.5585310459136963, 0.76114422082901] / acc : 0.740817872018175 / r2 : 0.41695637183989975
0.0005 > loss : [0.5947939157485962, 0.7480637431144714] / acc : 0.7234862827441396 / r2 : 0.38684189027220584
0.0001 > loss : [0.667610228061676, 0.723950982093811] / acc : 0.7079446490654366 / r2 : 0.31173715504192107




'''