import numpy as np
import time
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, StratifiedKFold, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

#1. 데이터
x, y = load_iris(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, stratify=y, random_state=3333)

n_splits=5
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=123)

parameters = [
    {'C':[1, 10, 100, 1000], 'kernel':['linear', 'sigmoid'], 'degree':[3,4,5]},  # 4*2*3=24번
    {'C':[1, 10, 100], 'kernel':['rbf'], 'gamma':[0.001, 0.0001]},  # 3*1*2=6번
    {'C':[1, 10, 100, 1000], 'kernel':['sigmoid'], 'gamma':[0.001, 0.0001, 0.0001], 'degree':[3,4]},  # 4*1*3*2=24번
]   # 24+6+24=54, 5(n_splits)*54(parameters)=270

#2. 모델 구성
model = GridSearchCV(SVC(), parameters, cv=kfold,
                     verbose=1,
                     refit=True,
                     n_jobs=-1,)

start = time.time()
model.fit(x_train, y_train)
end = time.time()

print('최적의 매개변수 :', model.best_estimator_)
print('최적의 파라미터 :', model.best_params_)

print('best_score :', model.best_score_)
print('model.score :', model.score(x_test, y_test))

y_predict = model.predict(x_test)
print('accuracy_score :', accuracy_score(y_test, y_predict))    # 이전과 차이를 보기위해

y_predict_best = model.best_estimator_.predict(x_test)
print('최적의 튠 ACC :', accuracy_score(y_test, y_predict_best))

print('time :', round(end - start, 2), '초')

'''
Fitting 5 folds for each of 54 candidates, totalling 270 fits
최적의 매개변수 : SVC(C=1, kernel='linear')
최적의 파라미터 : {'C': 1, 'degree': 3, 'kernel': 'linear'}
best_score : 0.9666666666666666
model.score : 1.0
accuracy_score : 1.0
최적의 튠 ACC : 1.0
time : 1.11 초
'''

import pandas as pd
# print(pd.DataFrame(model.cv_results_))
'''
    mean_fit_time  std_fit_time  mean_score_time  std_score_time  param_C  param_degree  ... split2_test_score  split3_test_score split4_test_score  mean_test_score  std_test_score  rank_test_score
0        0.001794  3.986128e-04         0.000598    4.881106e-04        1           3.0  ...          1.000000           0.916667          0.958333         0.966667        0.031180                1
1        0.002592  7.971766e-04         0.001395    7.976056e-04        1           3.0  ...          0.041667           0.041667          0.125000         0.075000        0.031180               43
2        0.001993  6.302988e-04         0.000798    3.989953e-04        1           4.0  ...          1.000000           0.916667          0.958333         0.966667        0.031180                1
...
51       0.000199  3.987312e-04         0.000399    4.883440e-04     1000           4.0  ...          1.000000           0.916667          0.916667         0.958333        0.037268                4
52       0.000399  4.882273e-04         0.000598    4.881883e-04     1000           4.0  ...          1.000000           0.916667          0.958333         0.958333        0.026352                4
53       0.000797  3.986836e-04         0.000199    3.986359e-04     1000           4.0  ...          1.000000           0.916667          0.958333         0.958333        0.026352                4
[54 rows x 17 columns]
'''
# print(pd.DataFrame(model.cv_results_).T)
'''
                                                          0                                           1   ...                                                 52                                                 53
mean_fit_time                                       0.001595                                    0.002193  ...                                           0.000598                                           0.000598
std_fit_time                                        0.000488                                    0.000399  ...                                           0.000488                                           0.000488
mean_score_time                                     0.000797                                    0.000996  ...                                           0.000399                                           0.000399
std_score_time                                      0.000399                                    0.000001  ...                                           0.000488                                           0.000488
param_C                                                    1                                           1  ...                                               1000                                               1000
param_degree                                             3.0                                         3.0  ...                                                4.0                                                4.0
param_kernel                                          linear                                     sigmoid  ...                                            sigmoid                                            sigmoid
param_gamma                                              NaN                                         NaN  ...                                             0.0001                                             0.0001
params             {'C': 1, 'degree': 3, 'kernel': 'linear'}  {'C': 1, 'degree': 3, 'kernel': 'sigmoid'}  ...  {'C': 1000, 'degree': 4, 'gamma': 0.0001, 'ker...  {'C': 1000, 'degree': 4, 'gamma': 0.0001, 'ker...
split0_test_score                                        1.0                                    0.083333  ...                                           0.958333                                           0.958333
split1_test_score                                   0.958333                                    0.083333  ...                                           0.958333                                           0.958333
split2_test_score                                        1.0                                    0.041667  ...                                                1.0                                                1.0
split3_test_score                                   0.916667                                    0.041667  ...                                           0.916667                                           0.916667
split4_test_score                                   0.958333                                       0.125  ...                                           0.958333                                           0.958333
mean_test_score                                     0.966667                                       0.075  ...                                           0.958333                                           0.958333
std_test_score                                       0.03118                                     0.03118  ...                                           0.026352                                           0.026352
rank_test_score                                            1                                          43  ...                                                  4                                                  4
[17 rows x 54 columns]
'''
# print(pd.DataFrame(model.cv_results_).sort_values('rank_test_score', ascending=True))
'''
    mean_fit_time  std_fit_time  mean_score_time  std_score_time  param_C  param_degree  ... split2_test_score  split3_test_score split4_test_score  mean_test_score  std_test_score  rank_test_score
0        0.001794  3.988270e-04         0.000997    8.844012e-07        1           3.0  ...          1.000000           0.916667          0.958333         0.966667        0.031180                1       
2        0.001993  6.299976e-04         0.000598    4.884609e-04        1           4.0  ...          1.000000           0.916667          0.958333         0.966667        0.031180                1       
4        0.001396  4.878971e-04         0.000798    3.988505e-04        1           5.0  ...          1.000000           0.916667          0.958333         0.966667        0.031180                1       
...
'''
print(pd.DataFrame(model.cv_results_).columns)
'''
Index(['mean_fit_time', 'std_fit_time', 'mean_score_time', 'std_score_time',
       'param_C', 'param_degree', 'param_kernel', 'param_gamma', 'params',
       'split0_test_score', 'split1_test_score', 'split2_test_score',
       'split3_test_score', 'split4_test_score', 'mean_test_score',
       'std_test_score', 'rank_test_score'],
      '''

path = 'C:\\ai5\\_save\\m15_GS_CV_01\\'
pd.DataFrame(model.cv_results_).sort_values('rank_test_score', ascending=True) \
    .to_csv(path + 'm15_GS_CV_results.csv')
