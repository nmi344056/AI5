import numpy as np
import time
import xgboost as xgb

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import r2_score, accuracy_score
from sklearn.datasets import load_boston

#1. 데이터
dataset = load_boston()
x = dataset.data
y = dataset.target

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=555)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

n_splits = 5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=123)

parameters = [
    {'n_jobs':[-1], 'n_estimators':[100, 500], 'max_depth':[None, 6, 10, 12], 'min_samples_leaf':[3, 10], 'learning_rate':[0.01, 0.001], 'max_leaf_nodes':[None, 3, 5, 7]},
    {'n_jobs':[-1], 'max_depth':[6, 9, 12], 'min_samples_leaf':[3, 9, 10], 'learning_rate':[0.01, 0.001], 'max_features':[None, 'sqrt', 'log2', 3, 5]},
    {'n_jobs':[-1], 'min_samples_leaf':[2, 5, 10], 'min_samples_split':[2, 5, 10], 'learning_rate':[0.01, 0.001]},
    {'n_jobs':[-1], 'min_samples_leaf':[2, 5, 10], 'learning_rate':[0.01, 0.001], 'max_leaf_nodes':[None, 3, 5, 7], 'max_features':[None, 'sqrt', 'log2', 3, 5]},
]   # 12+16+16+4=48

#2. 모델 구성
model = RandomizedSearchCV(xgb.XGBRegressor(), parameters, cv=kfold,
                     verbose=1,
                     refit=True,
                     n_jobs=-1,
                     n_iter=10,
                     random_state=3333,
                     )

start = time.time()
model.fit(x_train, y_train)
end = time.time()

print('최적의 파라미터 :', model.best_params_)
print('model.score :', model.score(x_test, y_test))

y_predict_best = model.best_estimator_.predict(x_test)
print('최적의 튠 r2 :', r2_score(y_test, y_predict_best))

print("time : ", round(end - start, 2), "초")

'''
loss :  24.317855834960938
r2 score :  0.7176152349757086
++++++++++++++++++++
dropout
loss :  15.876220703125
r2 score :  0.8156414819858147

GridSearchCV
최적의 파라미터 : {'max_depth': 8, 'min_samples_leaf': 3, 'n_jobs': -1}
model.score : 0.8488079955419329
최적의 튠 r2 : 0.8488079955419329
time :  5.12 초

RandomizedSearchCV
Fitting 5 folds for each of 10 candidates, totalling 50 fits
최적의 파라미터 : {'n_jobs': -1, 'n_estimators': 500, 'min_samples_leaf': 10, 'max_leaf_nodes': 3, 'max_depth': 10, 'learning_rate': 0.01}
model.score : 0.8635101754312281
최적의 튠 r2 : 0.8635101754312281
time :  5.41 초

'''
