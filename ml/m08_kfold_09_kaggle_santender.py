# https://www.kaggle.com/competitions/santander-customer-transaction-prediction

import numpy as np
import pandas as pd
import time
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC, SVR
import xgboost as xgb

#1. 데이터
path = "C:/ai5/_data/kaggle/santander-customer-transaction-prediction/"

train_csv = pd.read_csv(path + "train.csv", index_col=0)
test_csv = pd.read_csv(path + "test.csv", index_col=0)
submission_csv = pd.read_csv(path + "sample_submission.csv", index_col=0)

# print(train_csv.isna().sum())   # 결측치 없음
# print(test_csv.isna().sum())   # 결측치 없음

x = train_csv.drop(['target'], axis=1)
y = train_csv['target']

# print(x.shape)  # (200000, 200)
# print(y.shape)  # (200000,)

# print(pd.value_counts(y, sort=True))    # 이진 분류
# 0    179902
# 1     20098

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
scores = cross_val_score(model, x, y, cv=kfold)
end = time.time()

print('acc :', scores, 'avg acc :', round(np.mean(scores), 4))
print("time : ", round(end - start, 2), "초")

'''
MinMaxScaler / loss : 0.23171450197696686 acc : 0.91
StandardScaler / loss : 0.240584596991539 acc : 0.91

xgb
acc : [0.905025 0.90685  0.904525 0.90495  0.908075] avg acc : 0.9059
time :  30.81 초

'''
