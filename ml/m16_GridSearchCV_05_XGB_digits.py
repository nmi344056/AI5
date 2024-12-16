import numpy as np
import time
import xgboost as xgb
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, GridSearchCV
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.svm import SVC, SVR
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from sklearn.metrics import r2_score, accuracy_score

#1. 데이터
x, y = load_digits(return_X_y=True)
print(x)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, shuffle=True, random_state=123)

scaler = RobustScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

n_splits = 5
# kfold = KFold(n_splits=n_splits, shuffle=True, random_state=123)
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=123)

parameters = [
    {'n_jobs':[-1], 'n_estimators':[100, 500], 'max_depth':[6, 10, 12], 'min_samples_leaf':[3, 10]},    # 2*3*2=12번
    {'n_jobs':[-1], 'max_depth':[6, 8, 10, 12], 'min_samples_leaf':[3, 5, 7, 10]},                      # 4*4=16번
    {'n_jobs':[-1], 'min_samples_leaf':[2, 3, 5, 10], 'min_samples_split':[2, 3, 5, 10]},               # 4*4=16번
    {'n_jobs':[-1], 'min_samples_leaf':[2, 3, 5, 10]},                                                  # 4=4번
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
[실습] accuracy :  1.0 이상
128 256 256 256 128 10 / train_size=0.9, random_state=6666 / epochs=100, batch_size=100

loss :  0.11226184666156769
accuracy :  0.978

SVC
acc : [0.98888889 0.98055556 0.97771588 0.99164345 0.99442897] avg acc : 0.9866
time :  0.16 초

StratifiedKFold
acc : [0.98333333 0.98611111 0.99164345 0.99164345 0.98328691] avg acc : 0.9872
time :  0.16 초

train_test_split
acc : [0.95833333 0.94791667 0.95818815 0.96864111 0.96515679] avg acc : 0.9596
cross_val_predict : 0.6638150983040673
time :  0.17 초

GridSearchCV
최적의 파라미터 : {'max_depth': 6, 'min_samples_leaf': 3, 'n_estimators': 500, 'n_jobs': -1}
model.score : 0.9638888888888889
최적의 튠 acc : 0.9638888888888889
time :  7.22 초

'''
