# https://www.kaggle.com/competitions/otto-group-product-classification-challenge/overview

import numpy as np
import pandas as pd
import time
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, GridSearchCV
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.svm import SVC, SVR
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from sklearn.metrics import r2_score, accuracy_score

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

parameters = [
    {'tree_method': ['gpu_hist'], 'n_jobs':[-1], 'n_estimators':[100, 500], 'max_depth':[6, 10, 12], 'min_samples_leaf':[3, 10]},    # 2*3*2=12번
    {'tree_method': ['gpu_hist'], 'n_jobs':[-1], 'max_depth':[6, 8, 10, 12], 'min_samples_leaf':[3, 5, 7, 10]},                      # 4*4=16번
    {'tree_method': ['gpu_hist'], 'n_jobs':[-1], 'min_samples_leaf':[2, 3, 5, 10], 'min_samples_split':[2, 3, 5, 10]},               # 4*4=16번
    {'tree_method': ['gpu_hist'], 'n_jobs':[-1], 'min_samples_leaf':[2, 3, 5, 10]},                                                  # 4=4번
]   # 12+16+16+4=48

#2. 모델 구성
model = GridSearchCV(xgb.XGBClassifier(), parameters, cv=kfold,
                     verbose=1,
                     refit=True,
                     n_jobs=-1,)

start = time.time()
model.fit(x_train, y_train)
end = time.time()

print('최적의 파라미터 :', model.best_params_)
print('model.score :', model.score(x_test, y_test))

y_predict_best = model.best_estimator_.predict(x_test)
print('최적의 튠 acc :', accuracy_score(y_test, y_predict_best))

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

GridSearchCV


'''
    