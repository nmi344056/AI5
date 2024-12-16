import numpy as np
import pandas as pd
import time
from sklearn.datasets import load_boston, fetch_california_housing, load_diabetes
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.utils import all_estimators

import sklearn as sk
import warnings
warnings.filterwarnings('ignore')   # warnings.warn( 없애기

#1. 데이터
boston = load_boston(return_X_y=True)
california = fetch_california_housing(return_X_y=True)
diabetes = load_diabetes(return_X_y=True)

datasets = [boston, california, diabetes]
data_name = ['보스턴', '캘리포니아', '당뇨병']

#2. 모델 구성
# all = all_estimators(type_filter='classifier')
all = all_estimators(type_filter='regressor')

n_splits = 5
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=333)

start = time.time()
for index, value in enumerate(datasets):
    x, y = value

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, train_size=0.8, shuffle=True, random_state=123)

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    best_name = 0
    best_score = 0

    for name, model in all:
        try:
            #2. 모델
            model = model()
            #3. 훈련, 평가
            scores = cross_val_score(model, x_train, y_train, cv=kfold)
            # print('==========', data_name[index], name, '==========')
            # print('acc :', scores, 'avg acc :', round(np.mean(scores), 4))

            # y_predict = cross_val_predict(model, x_test, y_test, cv=kfold)
            # acc = accuracy_score(y_test, y_predict)
            
            if np.mean(scores) > best_score:
                best_name = name
                best_score = np.mean(scores)
            
        except:
            continue
            # print(name, '는 예외처리')

    print('==========', data_name[index], '==========')
    print('best model :', best_name, round(best_score, 4))

end = time.time()
print('time :', round(end-start,2), '초')

'''
========== 보스턴 ==========
best model : 0 0
========== 캘리포니아 ==========
best model : 0 0
========== 당뇨병 ==========
best model : ElasticNetCV 0.4685
time : 5.29 초

'''
