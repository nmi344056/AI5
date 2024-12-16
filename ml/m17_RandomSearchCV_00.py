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
