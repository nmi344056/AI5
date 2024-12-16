# https://www.kaggle.com/competitions/bike-sharing-demand/data?select=train.csv

import numpy as np
import pandas as pd
import time
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.svm import SVC, SVR
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from sklearn.metrics import r2_score, accuracy_score

#1. 데이터
path = "C:\\ai5\\_data\\kaggle\\bike-sharing-demand\\"
# path = "C://ai5//_data//bike-sharing-demand//"
# path = "C://ai5/_data/bike-sharing-demand/"

train_csv = pd.read_csv(path + "train.csv", index_col=0)
test_csv = pd.read_csv(path + "test.csv", index_col=0)
sampleSubmission = pd.read_csv(path + "sampleSubmission.csv", index_col=0)

# print(train_csv.shape)              # (10886, 11)
# print(test_csv.shape)               # (6493, 8)
# print(sampleSubmission.shape)       # (6493, 1)

# print(train_csv.columns)
# Index(['season', 'holiday', 'workingday', 'weather', 'temp', 'atemp',
#        'humidity', 'windspeed', 'casual', 'registered', 'count'],    
#       dtype='object')

# print(train_csv.info())             # 결측치가 없다
# print(test_csv.info())              # 결측치가 없다
# print(train_csv.describe())

########## 결측치 확인 ##########
# print(train_csv.isna().sum())       # 0
# print(train_csv.isnull().sum())     # 0
# print(test_csv.isna().sum())        # 0
# print(test_csv.isnull().sum())      # 0

########## x와 y를 분리 ##########
x = train_csv.drop(['casual', 'registered', 'count'], axis=1)
# print(x)                            # [10886 rows x 8 columns]
y = train_csv['count']
# print(y)
# print(y.shape)                      # (10886,)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, shuffle=True, random_state=123)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

n_splits = 5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=123)

#2. 모델 구성
model = SVR()

#3. 훈련
start = time.time()
scores = cross_val_score(model, x_train, y_train, cv=kfold)
end = time.time()

print('acc :', scores, 'avg acc :', round(np.mean(scores), 4))

y_predict = cross_val_predict(model, x_test, y_test)
r2 = r2_score(y_test, y_predict)
print('cross_val_predict :', r2)

print("time : ", round(end - start, 2), "초")

'''
128 64 32 32 32 1 / train_size=0.85, random_state=111 / epochs=500, batch_size=100

loss :  21271.451171875
re score :  0.35337467674572753

SVR
acc : [0.19290674 0.18965429 0.19277704 0.2131451  0.20084338] avg acc : 0.1979
time :  11.47 초

train_test_split
acc : [0.18561331 0.21933573 0.23775875 0.2234596  0.20638356] avg acc : 0.2145
cross_val_predict : 0.10036461532413143
time :  7.53 초

'''
