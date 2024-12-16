import numpy as np
import time
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, StratifiedKFold, GridSearchCV, RandomizedSearchCV
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
# model = GridSearchCV(SVC(), parameters, cv=kfold,
#                      verbose=1,
#                      refit=True,
#                      n_jobs=-1,)

model = RandomizedSearchCV(SVC(), parameters, cv=kfold,
                     verbose=1,
                     refit=True,
                     n_jobs=-1,
                     n_iter=10,
                     random_state=3333,
                     )

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
GridSearchCV
Fitting 5 folds for each of 54 candidates, totalling 270 fits
최적의 매개변수 : SVC(C=1, kernel='linear')
최적의 파라미터 : {'C': 1, 'degree': 3, 'kernel': 'linear'}
best_score : 0.9666666666666666
model.score : 1.0
accuracy_score : 1.0
최적의 튠 ACC : 1.0
time : 1.11 초

RandomizedSearchCV
Fitting 5 folds for each of 10 candidates, totalling 50 fits
최적의 매개변수 : SVC(C=1000, degree=4, gamma=0.001, kernel='sigmoid')
최적의 파라미터 : {'kernel': 'sigmoid', 'gamma': 0.001, 'degree': 4, 'C': 1000}
best_score : 0.9583333333333334
model.score : 1.0
accuracy_score : 1.0
최적의 튠 ACC : 1.0
time : 1.05 초
'''

import pandas as pd
# print(pd.DataFrame(model.cv_results_))
'''
   mean_fit_time  std_fit_time  mean_score_time  std_score_time param_kernel  param_gamma  ...  split2_test_score  split3_test_score split4_test_score  mean_test_score  std_test_score  rank_test_score
0       0.002392  4.883640e-04         0.001196    3.981831e-04          rbf       0.0001  ...           0.916667           0.833333          0.875000         0.891667        0.062361                7
1       0.002392  4.886561e-04         0.001595    4.885582e-04      sigmoid          NaN  ...           0.000000           0.041667          0.000000         0.025000        0.020412               10
2       0.001993  9.655217e-07         0.000996    6.910027e-07       linear          NaN  ...           1.000000           0.916667          0.958333         0.966667        0.031180                1
...
7       0.001594  4.884610e-04         0.000598    4.883832e-04          rbf       0.0001  ...           0.916667           0.833333          0.916667         0.908333        0.055277                4
8       0.001395  4.880717e-04         0.000598    4.882274e-04       linear          NaN  ...           1.000000           0.875000          0.916667         0.933333        0.056519                2
9       0.001395  4.882079e-04         0.000797    3.986120e-04      sigmoid       0.0001  ...           0.916667           0.833333          0.875000         0.891667        0.062361                7
[10 rows x 17 columns]
'''
# print(pd.DataFrame(model.cv_results_).T)
'''
                                                            0                                             1  ...                                           8                                                  9
mean_fit_time                                        0.001795                                      0.002392  ...                                    0.001395                                           0.001794
std_fit_time                                         0.000398                                      0.000488  ...                                    0.000489                                           0.000399
mean_score_time                                      0.001395                                      0.000798  ...                                    0.000399                                           0.000797
std_score_time                                       0.000488                                      0.000399  ...                                    0.000488                                           0.000399
param_kernel                                              rbf                                       sigmoid  ...                                      linear                                            sigmoid
param_gamma                                            0.0001                                           NaN  ...                                         NaN                                             0.0001
param_C                                                     1                                           100  ...                                          10                                                  1
param_degree                                              NaN                                           5.0  ...                                         3.0                                                4.0
params             {'kernel': 'rbf', 'gamma': 0.0001, 'C': 1}  {'kernel': 'sigmoid', 'degree': 5, 'C': 100}  ...  {'kernel': 'linear', 'degree': 3, 'C': 10}  {'kernel': 'sigmoid', 'gamma': 0.0001, 'degree...
split0_test_score                                    0.833333                                      0.041667  ...                                         1.0                                           0.833333
split1_test_score                                         1.0                                      0.041667  ...                                       0.875                                                1.0
split2_test_score                                    0.916667                                           0.0  ...                                         1.0                                           0.916667
split3_test_score                                    0.833333                                      0.041667  ...                                       0.875                                           0.833333
split4_test_score                                       0.875                                           0.0  ...                                    0.916667                                              0.875
mean_test_score                                      0.891667                                         0.025  ...                                    0.933333                                           0.891667
std_test_score                                       0.062361                                      0.020412  ...                                    0.056519                                           0.062361
rank_test_score                                             7                                            10  ...                                           2                                                  7
[17 rows x 10 columns]
'''
# print(pd.DataFrame(model.cv_results_).sort_values('rank_test_score', ascending=True))
'''
   mean_fit_time  std_fit_time  mean_score_time  std_score_time param_kernel  param_gamma  ...  split2_test_score  split3_test_score split4_test_score  mean_test_score  std_test_score  rank_test_score
2       0.001595      0.000488         0.000996    6.298463e-04       linear          NaN  ...           1.000000           0.916667          0.958333         0.966667        0.031180                1
5       0.001196      0.000399         0.000997    5.352484e-07       linear          NaN  ...           1.000000           0.875000          0.916667         0.933333        0.056519                2
8       0.001197      0.000399         0.000598    4.880326e-04       linear          NaN  ...           1.000000           0.875000          0.916667         0.933333        0.056519                2
...
3       0.001196      0.000399         0.000598    4.883052e-04      sigmoid       0.0001  ...           0.916667           0.833333          0.875000         0.891667        0.062361                7
9       0.001395      0.000488         0.000797    3.984929e-04      sigmoid       0.0001  ...           0.916667           0.833333          0.875000         0.891667        0.062361                7
1       0.002592      0.000489         0.000997    4.862804e-07      sigmoid          NaN  ...           0.000000           0.041667          0.000000         0.025000        0.020412               10
[10 rows x 17 columns]
'''
print(pd.DataFrame(model.cv_results_).columns)
'''
Index(['mean_fit_time', 'std_fit_time', 'mean_score_time', 'std_score_time',
       'param_kernel', 'param_gamma', 'param_C', 'param_degree', 'params',
       'split0_test_score', 'split1_test_score', 'split2_test_score',
       'split3_test_score', 'split4_test_score', 'mean_test_score',
       'std_test_score', 'rank_test_score'],
'''

path = 'C:\\ai5\\_save\\m17_RS_CV_01\\'
pd.DataFrame(model.cv_results_).sort_values('rank_test_score', ascending=True) \
    .to_csv(path + 'm17_RS_CV_results.csv')
