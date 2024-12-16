# https://www.kaggle.com/competitions/playground-series-s4e1

import numpy as np
import pandas as pd
import time
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.svm import SVC, SVR
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from sklearn.metrics import r2_score, accuracy_score
import xgboost as xgb

#1. 데이터
path = "C:\\ai5\\_data\\kaggle\\playground-series-s4e1\\"

train_csv = pd.read_csv(path + "train.csv", index_col=0)
test_csv = pd.read_csv(path + "test.csv", index_col=0)
mission_csv = pd.read_csv(path + "sample_submission.csv", index_col=0)

# print(train_csv.shape)      # (165034, 13)
# print(test_csv.shape)       # (110023, 12)
# print(mission_csv.shape)    # (110023, 1)

# print(train_csv.columns)
# Index(['CustomerId', 'Surname', 'CreditScore', 'Geography', 'Gender', 'Age',
#        'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember',
#        'EstimatedSalary', 'Exited'],

# print(train_csv.isnull().sum())     # 결측치가 없다
# print(test_csv.isnull().sum())

encoder = LabelEncoder()
train_csv['Geography'] = encoder.fit_transform(train_csv['Geography'])
test_csv['Geography'] = encoder.fit_transform(test_csv['Geography'])
train_csv['Gender'] = encoder.fit_transform(train_csv['Gender'])
test_csv['Gender'] = encoder.fit_transform(test_csv['Gender'])

train_csv = train_csv.drop(['CustomerId', 'Surname'], axis=1)
test_csv = test_csv.drop(['CustomerId', 'Surname'], axis=1)

###############################################
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

x = train_csv.drop(['Exited'], axis=1)
print(x)                            # [165034 rows x 10 columns]
y = train_csv['Exited']
print(y.shape)                      # (165034,)

# print(np.unique(y, return_counts=True))     
# (array([0, 1], dtype=int64), array([130113,  34921], dtype=int64))
# print(pd.DataFrame(y).value_counts())
# 0         130113
# 1          34921

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, shuffle=True, random_state=123)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

n_splits = 5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=123)

#2. 모델 구성
# model = SVR()                 # 너무 오래 걸린다?

model = xgb.XGBClassifier(
    n_estimators=300,           # epochs
    learning_rate=0.1,
    max_depth=2,                # 2
    random_state=123,           # 123
    use_label_encoder=False,
    eval_metric='mlogloss',
    gamma=2,                    # 2
    # colsample_bytree=0.7,
)

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
32 64 128 128 128 64 32 1 / train_size=0.9, random_state=1866 / patience=10 / epochs=1000, batch_size=8
loss :  0.3232889473438263 / accuracy :  0.866 > 0.75422

loss :  0.32732534408569336
accuracy :  0.863

SVR
너무 오래 걸린다? (1시간 정도 기다린 듯, 메모리 사용 확인 시)

xgb
acc : [0.86299876 0.86642227 0.86427122 0.86575575 0.86196449] avg acc : 0.8643
time :  1.54 초

train_test_split
acc : [0.86597743 0.8649928  0.86282901 0.864306   0.86532854] avg acc : 0.8647
cross_val_predict : 0.18566329694493988
time :  0.98 초

'''
