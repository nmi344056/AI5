import numpy as np
import time
import xgboost as xgb
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, GridSearchCV
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.svm import SVC, SVR
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from sklearn.metrics import r2_score, accuracy_score

#1. 데이터
x, y = load_diabetes(return_X_y=True)
# print(x)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, shuffle=True, random_state=123)

scaler = MaxAbsScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

n_splits = 5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=123)

parameters = [
    {'n_jobs':[-1], 'n_estimators':[100, 500], 'max_depth':[6, 10, 12], 'min_samples_leaf':[3, 10]},    # 2*3*2=12번
    {'n_jobs':[-1], 'max_depth':[6, 8, 10, 12], 'min_samples_leaf':[3, 5, 7, 10]},                      # 4*4=16번
    {'n_jobs':[-1], 'min_samples_leaf':[2, 3, 5, 10], 'min_samples_split':[2, 3, 5, 10]},               # 4*4=16번
    {'n_jobs':[-1], 'min_samples_leaf':[2, 3, 5, 10]},                                                  # 4=4번
]   # 12+16+16+4=48

#2. 모델 구성
model = GridSearchCV(xgb.XGBRegressor(), parameters, cv=kfold,
                     verbose=1,
                     refit=True,
                     n_jobs=-1,)

start = time.time()
model.fit(x_train, y_train)
end = time.time()

print('최적의 파라미터 :', model.best_params_)
print('model.score :', model.score(x_test, y_test))

y_predict_best = model.best_estimator_.predict(x_test)
print('최적의 튠 r2 :', r2_score(y_test, y_predict_best))

print("time : ", round(end - start, 2), "초")

'''
loss :  2197.28173828125
r2 sorce :  0.596755557322737

loss :  2156.3740234375
r2 sorce :  0.6042629302834428

SVR
acc : [0.18093916 0.15563965 0.15939131 0.15210134 0.14966275] avg acc : 0.1595
time :  0.03 초

train_test_split
acc : [0.04610951 0.04296415 0.11195374 0.08379547 0.08982703] avg acc : 0.0749
cross_val_predict : -0.03948304536503011
time :  0.02 초

GridSearchCV
최적의 파라미터 : {'max_depth': 6, 'min_samples_leaf': 3, 'n_estimators': 100, 'n_jobs': -1}
model.score : 0.39065385219018145
최적의 튠 r2 : 0.39065385219018145
time :  4.41 초

'''
