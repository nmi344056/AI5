# https://www.kaggle.com/competitions/otto-group-product-classification-challenge/overview

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from sklearn.metrics import r2_score, accuracy_score
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd
import time

np.random.seed(337)         # numpy seed 고정
import tensorflow as tf
tf.random.set_seed(337)     # tensorflow seed 고정
import random as rn
rn.seed(337)                # python seed 고정

#1. 데이터
path = "C:\\ai5\\_data\\kaggle\\otto-group-product-classification-challenge\\"

train_csv = pd.read_csv(path + "train.csv", index_col=0)
test_csv = pd.read_csv(path + "test.csv", index_col=0)
submission_csv = pd.read_csv(path + "sampleSubmission.csv", index_col=0)

print(train_csv.isna().sum())       # 결측치 없음
print(test_csv.isna().sum())        # 결측치 없음

print(train_csv)                    # target이 Class_1 ... Class_9

encoder = LabelEncoder()
train_csv['target'] = encoder.fit_transform(train_csv['target'])    # 라벨링

print(train_csv)                    # target이 0 ... 8

x = train_csv.drop(['target'], axis=1)
y = train_csv['target']

print(x.shape)                      # (61878, 93), =input_dim
print(y.shape)                      # (61878,)

y_ohe = pd.get_dummies(y)           # OneHot

print(y_ohe.shape)                  # (61878, 9) ,9의 추가를 통해 OneHot 학인, =output_dim

print(pd.value_counts(y, sort=False))   # 동일하게 나온다, 컴퓨터가 인식할때는 숫자로 인식해서?

x_train, x_test, y_train, y_test = train_test_split(x, y_ohe, train_size=0.9, random_state=3, stratify=y)

from sklearn.preprocessing import MinMaxScaler, StandardScaler
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
test_csv = scaler.transform(test_csv)

lr = [0.1, 0.01, 0.005, 0.001, 0.0005, 0.0001]
results = []

from tensorflow.keras.optimizers import Adam
for i in range(len(lr)): 

    #2. 모델 구성
    model = Sequential()
    model.add(Dense(128, activation='relu', input_shape=(x_train.shape[1],)))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(1024, activation='relu'))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(9, activation='softmax'))

    #3. 컴파일, 훈련
    model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=lr[i]), metrics=['accuracy'])

    es = EarlyStopping(monitor='val_loss', mode='min', patience=10, verbose=1, restore_best_weights=True,)

    ########## mcp 세이브 파일명 만들기 시작 ##########
    import datetime
    date = datetime.datetime.now()
    date = date.strftime("%m%d_%H%M")

    path = './_save/keras68/'
    filename = '{epoch:04d}_valloss_{val_loss:.4f}.hdf5'
    filepath = "".join([path, 'k68_04_date_', str(i+1), '_', date, '_epo_', filename])

    ########## mcp 세이브 파일명 만들기 끝 ##########

    mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1, save_best_only=True, filepath=filepath)

    start = time.time()
    hist = model.fit(x_train, y_train, epochs=500, batch_size=60, validation_split=0.2, callbacks=[es, mcp])
    end = time.time()

    #4. 평가, 예측
    loss = model.evaluate(x_test, y_test, verbose=1)
    # print('loss :', loss)

    y_predict = model.predict(x_test)
    r2 = r2_score(y_test, y_predict)

    # # csv 파일 만들기 #
    # y_submit = model.predict(test_csv)
    # y_submit = np.round(y_submit)

    # submission_csv[['Class_1', 'Class_2', 'Class_3', 'Class_4', 'Class_5', 'Class_6', 'Class_7', 'Class_8', 'Class_9']] = y_submit  # 변경
    # submission_csv.to_csv(path + "sampleSubmission_0724_20.csv")

    # results.append([i+1, num[i], round(end-start,2), round(loss[1], 3)])

    print('{0} > loss : {1} / r2 : {2}'.format(lr[i], loss, r2))

# for a in results:
#     print('결과', a[0])
#     print('PCA :', a[1])
#     print('time :', a[2],'초')
#     print('accuracy :', a[3])
#     print('')

'''
결과 1
PCA : 62
time : 3.3 초
accuracy : 0.739

결과 2
PCA : 82
time : 2.72 초
accuracy : 0.753

결과 3
PCA : 91
time : 3.14 초
accuracy : 0.748

결과 4
PCA : 93
time : 3.1 초
accuracy : 0.742
'''
