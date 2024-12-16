import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
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
x, y = load_iris(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, stratify=y, shuffle=True, random_state=123
)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델 구성
all = all_estimators(type_filter='classifier')
# all = all_estimators(type_filter='regressor')

# print('sk 버전 :', sk.__version__)    # 1.5.1
# print('all Algorithms :', all)
# print('모델의 갯수 :', len(all))      # 43 (sk 버전마다 다르다)

n_splits = 5
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=333)

for name, model in all:
    try:
        #2. 모델
        model = model()
        #3. 훈련, 평가
        scores = cross_val_score(model, x_train, y_train, cv=kfold)
        print('==========', name, '==========')
        print('acc :', scores, 'avg acc :', round(np.mean(scores), 4))

        y_predict = cross_val_predict(model, x_test, y_test, cv=kfold)
        acc = accuracy_score(y_test, y_predict)
        print('cross_val_predict :', acc)
    except:
        print(name, '는 예외처리')

'''
========== AdaBoostClassifier ==========
acc : [0.95833333 0.875      0.91666667 0.95833333 0.95833333] avg acc : 0.9333
cross_val_predict : 0.9666666666666667
========== BaggingClassifier ==========
acc : [0.95833333 0.875      0.95833333 0.95833333 0.95833333] avg acc : 0.9417
cross_val_predict : 0.9666666666666667
========== BernoulliNB ==========
acc : [0.75       0.70833333 0.79166667 0.79166667 0.83333333] avg acc : 0.775
cross_val_predict : 0.6333333333333333
========== CalibratedClassifierCV ==========
acc : [0.875      0.83333333 0.95833333 0.91666667 0.875     ] avg acc : 0.8917
cross_val_predict : 0.8666666666666667
CategoricalNB 는 예외처리
ClassifierChain 는 예외처리
ComplementNB 는 예외처리
========== DecisionTreeClassifier ==========
acc : [0.95833333 0.875      0.91666667 0.95833333 0.95833333] avg acc : 0.9333
cross_val_predict : 0.9666666666666667
========== DummyClassifier ==========
acc : [0.33333333 0.33333333 0.33333333 0.33333333 0.33333333] avg acc : 0.3333
cross_val_predict : 0.3333333333333333
========== ExtraTreeClassifier ==========
acc : [0.95833333 0.91666667 0.95833333 0.95833333 0.95833333] avg acc : 0.95
cross_val_predict : 0.9666666666666667
========== ExtraTreesClassifier ==========
acc : [0.95833333 0.91666667 0.95833333 1.         1.        ] avg acc : 0.9667
cross_val_predict : 1.0
FixedThresholdClassifier 는 예외처리
========== GaussianNB ==========
acc : [0.91666667 0.91666667 0.95833333 1.         0.95833333] avg acc : 0.95
cross_val_predict : 0.8666666666666667
========== GaussianProcessClassifier ==========
acc : [0.91666667 0.95833333 0.95833333 1.         0.91666667] avg acc : 0.95
cross_val_predict : 0.9666666666666667
========== GradientBoostingClassifier ==========
acc : [0.91666667 0.91666667 1.         0.95833333 0.95833333] avg acc : 0.95
cross_val_predict : 0.9
========== HistGradientBoostingClassifier ==========
acc : [0.95833333 0.91666667 0.95833333 0.95833333 0.95833333] avg acc : 0.95
cross_val_predict : 0.3333333333333333
========== KNeighborsClassifier ==========
acc : [0.91666667 1.         0.91666667 1.         1.        ] avg acc : 0.9667
cross_val_predict : 0.9333333333333333
========== LabelPropagation ==========
acc : [0.91666667 0.91666667 0.91666667 0.95833333 0.95833333] avg acc : 0.9333
cross_val_predict : 0.9
========== LabelSpreading ==========
acc : [0.91666667 0.91666667 0.91666667 0.95833333 0.95833333] avg acc : 0.9333
cross_val_predict : 0.9
========== LinearDiscriminantAnalysis ==========
acc : [0.95833333 0.91666667 1.         1.         1.        ] avg acc : 0.975
cross_val_predict : 1.0
========== LinearSVC ==========
acc : [0.95833333 0.91666667 0.95833333 0.95833333 0.95833333] avg acc : 0.95
cross_val_predict : 0.8666666666666667
========== LogisticRegression ==========
acc : [0.95833333 0.91666667 0.95833333 1.         1.        ] avg acc : 0.9667
cross_val_predict : 0.9
========== LogisticRegressionCV ==========
acc : [0.95833333 0.95833333 0.95833333 1.         1.        ] avg acc : 0.975
cross_val_predict : 0.9666666666666667
========== MLPClassifier ==========
acc : [0.91666667 0.91666667 0.95833333 1.         1.        ] avg acc : 0.9583
cross_val_predict : 0.9
MultiOutputClassifier 는 예외처리
MultinomialNB 는 예외처리
========== NearestCentroid ==========
acc : [0.91666667 0.83333333 0.875      0.95833333 0.83333333] avg acc : 0.8833
cross_val_predict : 0.7666666666666667
========== NuSVC ==========
acc : [0.91666667 0.95833333 0.95833333 1.         1.        ] avg acc : 0.9667
cross_val_predict : 0.9666666666666667
OneVsOneClassifier 는 예외처리
OneVsRestClassifier 는 예외처리
OutputCodeClassifier 는 예외처리
========== PassiveAggressiveClassifier ==========
acc : [0.91666667 0.83333333 0.91666667 0.875      0.95833333] avg acc : 0.9
cross_val_predict : 0.9333333333333333
========== Perceptron ==========
acc : [0.79166667 0.66666667 0.95833333 0.79166667 0.79166667] avg acc : 0.8
cross_val_predict : 0.7666666666666667
========== QuadraticDiscriminantAnalysis ==========
acc : [0.95833333 0.91666667 1.         1.         1.        ] avg acc : 0.975
cross_val_predict : 0.9
========== RadiusNeighborsClassifier ==========
acc : [0.95833333        nan 0.95833333        nan        nan] avg acc : nan
RadiusNeighborsClassifier 는 예외처리
========== RandomForestClassifier ==========
acc : [0.95833333 0.91666667 0.95833333 0.95833333 0.95833333] avg acc : 0.95
cross_val_predict : 0.9666666666666667
========== RidgeClassifier ==========
acc : [0.83333333 0.70833333 0.83333333 0.91666667 0.875     ] avg acc : 0.8333
cross_val_predict : 0.8666666666666667
========== RidgeClassifierCV ==========
acc : [0.83333333 0.70833333 0.79166667 0.91666667 0.875     ] avg acc : 0.825
cross_val_predict : 0.8333333333333334
========== SGDClassifier ==========
acc : [0.83333333 0.95833333 0.95833333 0.95833333 0.91666667] avg acc : 0.925
cross_val_predict : 0.8333333333333334
========== SVC ==========
acc : [0.95833333 0.95833333 0.95833333 1.         1.        ] avg acc : 0.975
cross_val_predict : 0.8666666666666667
StackingClassifier 는 예외처리
TunedThresholdClassifierCV 는 예외처리
VotingClassifier 는 예외처리
'''
