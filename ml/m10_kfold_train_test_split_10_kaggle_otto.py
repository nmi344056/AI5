# https://www.kaggle.com/competitions/otto-group-product-classification-challenge/overview

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
path = "C:\\ai5\\_data\\kaggle\\otto-group-product-classification-challenge\\"

train_csv = pd.read_csv(path + "train.csv", index_col=0)
test_csv = pd.read_csv(path + "test.csv", index_col=0)
submission_csv = pd.read_csv(path + "sampleSubmission.csv", index_col=0)

# print(train_csv.isna().sum())       # 결측치 없음
# print(test_csv.isna().sum())        # 결측치 없음
# print(train_csv)                    # target이 Class_1 ... Class_9

encoder = LabelEncoder()
train_csv['target'] = encoder.fit_transform(train_csv['target'])    # 라벨링

# print(train_csv)                    # target이 0 ... 8

x = train_csv.drop(['target'], axis=1)
y = train_csv['target']

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, shuffle=True, random_state=123)

scaler = RobustScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

n_splits = 5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=123)

#2. 모델 구성
# model = SVR()

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
loss : [0.5208882689476013, 0.8039754629135132]
acc : 0.804

xgb
acc : [0.77884615 0.77973497 0.77900776 0.77010101 0.77777778] avg acc : 0.7771
time :  9.07 초

acc : [0.77355823 0.76850823 0.78070707 0.77727273 0.77808081] avg acc : 0.7756
cross_val_predict : 0.6253324721981235
time :  7.37 초

'''
