# https://www.kaggle.com/competitions/santander-customer-transaction-prediction

import numpy as np
import pandas as pd
import time
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from sklearn.metrics import r2_score, accuracy_score

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

from sklearn.preprocessing import MinMaxScaler, StandardScaler
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

from sklearn.decomposition import PCA
pca = PCA(n_components=len(np.array(x)[1]))
x_train = pca.fit_transform(x_train)
x_test = pca.transform(x_test)

evr_cumsum = np.cumsum(pca.explained_variance_ratio_)

num = [np.argmax(evr_cumsum>=0.95)+1, np.argmax(evr_cumsum>=0.99)+1, np.argmax(evr_cumsum>=0.999)+1, np.argmax(evr_cumsum)+1]
results = []

for i in range(0, len(num), 1):
    pca = PCA(n_components=num[i])
    x_train1 = pca.fit_transform(x_train)
    x_test1 = pca.transform(x_test)

    #2. 모델 구성
    model = Sequential()
    model.add(Dense(512, input_shape=(num[i],), activation='relu'))
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
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

    es = EarlyStopping(monitor='val_loss', mode='min', patience=100, verbose=1, restore_best_weights=True,)

    ########## mcp 세이브 파일명 만들기 시작 ##########
    import datetime
    date = datetime.datetime.now()
    date = date.strftime("%m%d_%H%M")

    path = './_save/m05/'
    filename = '{epoch:04d}_valloss_{val_loss:.4f}.hdf5'
    filepath = "".join([path, 'm05_12_date_', str(i+1), '_', date, '_epo_', filename])

    ########## mcp 세이브 파일명 만들기 끝 ##########

    mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1, save_best_only=True, filepath=filepath)

    start = time.time()
    hist = model.fit(x_train1, y_train, epochs=500, batch_size=5000, validation_split=0.2, callbacks=[es, mcp])
    end = time.time()

    #4. 평가, 예측
    loss = model.evaluate(x_test1, y_test)
    # print('loss :', loss[0])
    # print('acc :', round(loss[1],2))

    y_predict1 = model.predict(x_test1)
    # r2 = r2_score(y_test,y_predict1)
    # print('r2 score :', r2)

    # accuracy_score = accuracy_score(y_test,np.round(y_predict))
    # print('acc_score :', accuracy_score)

    # ### csv 파일 만들기 ###
    # y_submit = model.predict(test_csv)
    # print(y_submit)

    # y_submit = np.round(y_submit)
    # print(y_submit)

    # submission_csv['target'] = y_submit
    # submission_csv.to_csv(path + "sampleSubmission_0724_1640.csv")

    # print(submission_csv['target'].value_counts())

    results.append([i+1, num[i], round(end-start,2), round(loss[1], 3)])

for a in results:
    print('결과', a[0])
    print('PCA :', a[1])
    print('time :', a[2],'초')
    print('accuracy :', a[3])
    print('')

'''
결과 1
PCA : 186
time : 1.26 초
accuracy : 0.9

결과 2
PCA : 197
time : 0.7 초
accuracy : 0.9

결과 3
PCA : 200
time : 0.86 초
accuracy : 0.9

결과 4
PCA : 200
time : 0.94 초
accuracy : 0.9
'''
