# https://www.kaggle.com/competitions/santander-customer-transaction-prediction

import numpy as np
import pandas as pd
import time
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from sklearn.metrics import r2_score, accuracy_score

np.random.seed(337)         # numpy seed 고정
import tensorflow as tf
tf.random.set_seed(337)     # tensorflow seed 고정
import random as rn
rn.seed(337)                # python seed 고정

#1. 데이터
path = "C:/ai5/_data/kaggle/santander-customer-transaction-prediction/"

train_csv = pd.read_csv(path + "train.csv", index_col=0)
test_csv = pd.read_csv(path + "test.csv", index_col=0)
submission_csv = pd.read_csv(path + "sample_submission.csv", index_col=0)

print(train_csv.isna().sum())   # 결측치 없음
print(test_csv.isna().sum())   # 결측치 없음

x = train_csv.drop(['target'], axis=1)
y = train_csv['target']

print(x.shape)  # (200000, 200)
print(y.shape)  # (200000,)

print(pd.value_counts(y, sort=True))    # 이진 분류
# 0    179902
# 1     20098

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=5233,
                                                    stratify=y)

scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
# scaler = RobustScaler()

x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
test_csv = scaler.transform(test_csv)

print(x_train)
print(np.min(x_train), np.max(x_train))     # 
print(np.min(x_test), np.max(x_test))       # 

lr = [0.1, 0.01, 0.005, 0.001, 0.0005, 0.0001]
results = []

for i in range(len(lr)):

    #2. 모델 구성
    model = Sequential()
    model.add(Dense(512, input_shape=(x_train.shape[1],), activation='relu'))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    #3. 컴파일, 훈련
    from tensorflow.keras.optimizers import Adam
    model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=lr[i]), metrics=['accuracy'])

    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
    es = EarlyStopping(monitor='val_loss', mode='min', patience=100, verbose=1, restore_best_weights=True,)
    rlr = ReduceLROnPlateau(monitor='val_loss', mode='auto', patience=15, verbose=1, factor=0.7)

    ########## mcp 세이브 파일명 만들기 시작 ##########
    import datetime
    date = datetime.datetime.now()
    date = date.strftime("%m%d_%H%M")

    path = './_save/keras69/'
    filename = '{epoch:04d}_valloss_{val_loss:.4f}.hdf5'
    filepath = "".join([path, 'k69_03_date_', str(i+1), '_', date, '_epo_', filename])

    ########## mcp 세이브 파일명 만들기 끝 ##########

    mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1, save_best_only=True, filepath=filepath)

    start = time.time()
    hist = model.fit(x_train, y_train, epochs=500, batch_size=16, validation_split=0.2, callbacks=[es, mcp])
    end = time.time()

    #4. 평가, 예측
    loss = model.evaluate(x_test, y_test)
    # print('loss :', loss[0])
    # print('acc :', round(loss[1],2))

    y_predict = model.predict(x_test)
    r2 = r2_score(y_test,y_predict)
    # print('r2 score :', r2)

    acc = accuracy_score(y_test,np.round(y_predict))
    # print('acc_score :', accuracy_score)

    # ### csv 파일 만들기 ###
    # y_submit = model.predict(test_csv)
    # print(y_submit)

    # y_submit = np.round(y_submit)
    # print(y_submit)

    # submission_csv['target'] = y_submit
    # submission_csv.to_csv(path + "sampleSubmission_0724_1640.csv")

    # print(submission_csv['target'].value_counts())

    print('{0} > loss : {1} / acc : {2} / r2 : {3}'.format(lr[i], loss, acc, r2))

    # print('결과', i+1)
    # print('learning_rate :',lr[i])
    # print('time :', round(end-start,2),'초')
    # print('loss, accuracy :', loss)
    # print('')

# results.append([i+1, lr[i], round(end-start,2), loss[1]])

# for a in results:
#     print('결과', a[0])
#     print('learning_rate :', a[1])
#     print('time :', a[2],'초')
#     print('accuracy :', a[3])
#     print('')

'''
결과 1
learning_rate : 0.1
time : 1.24 초
accuracy : 0.8995000123977661

결과 2
learning_rate : 0.01
time : 0.64 초
accuracy : 0.8995000123977661

결과 3
learning_rate : 0.005
time : 0.7 초
accuracy : 0.8995000123977661

결과 4
learning_rate : 0.001
time : 0.78 초
accuracy : 0.8995000123977661

결과 5
learning_rate : 0.0005
time : 0.71 초
accuracy : 0.8995000123977661

결과 6
learning_rate : 0.0001
time : 0.65 초
accuracy : 0.8995000123977661

learning_rate : 0.1
time : 2.26 초
accuracy : 0.8995000123977661

learning_rate : 0.01
time : 1.8 초
accuracy : 0.8979499936103821

결과 4
learning_rate : 0.001
time : 1.85 초
accuracy : 0.9129999876022339




'''
