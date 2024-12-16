import numpy as np
import pandas as pd
import time
from sklearn.datasets import load_iris, load_breast_cancer, load_wine, load_digits
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
iris = load_iris(return_X_y=True)
cancer = load_breast_cancer(return_X_y=True)
wine = load_wine(return_X_y=True)
digits = load_digits(return_X_y=True)

datasets = [iris, cancer, wine, digits]
data_name = ['아이리스', '캔서', '와인', '디지트']

#2. 모델 구성
all = all_estimators(type_filter='classifier')
# all = all_estimators(type_filter='regressor')

n_splits = 5
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=333)

start = time.time()
for index, value in enumerate(datasets):
    x, y = value

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, train_size=0.8, stratify=y, shuffle=True, random_state=123)

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    for name, model in all:
        try:
            #2. 모델
            model = model()
            #3. 훈련, 평가
            scores = cross_val_score(model, x_train, y_train, cv=kfold)
            print('==========', data_name[index], name, '==========')
            print('acc :', scores, 'avg acc :', round(np.mean(scores), 4))

            y_predict = cross_val_predict(model, x_test, y_test, cv=kfold)
            acc = accuracy_score(y_test, y_predict)
            print('cross_val_predict :', acc)
        except:
            print(name, '는 예외처리')
end = time.time()
print('time :', round(end-start,2), '초')

'''
========== 아이리스 AdaBoostClassifier ==========
acc : [0.95833333 0.875      0.91666667 0.95833333 0.95833333] avg acc : 0.9333
cross_val_predict : 0.9666666666666667
========== 아이리스 BaggingClassifier ==========
acc : [0.95833333 0.95833333 0.95833333 0.95833333 0.95833333] avg acc : 0.9583
cross_val_predict : 0.9666666666666667
========== 아이리스 BernoulliNB ==========
acc : [0.75       0.70833333 0.79166667 0.79166667 0.83333333] avg acc : 0.775
cross_val_predict : 0.6333333333333333
========== 아이리스 CalibratedClassifierCV ==========
acc : [0.875      0.83333333 0.95833333 0.91666667 0.875     ] avg acc : 0.8917
cross_val_predict : 0.8666666666666667
CategoricalNB 는 예외처리
ClassifierChain 는 예외처리
ComplementNB 는 예외처리
========== 아이리스 DecisionTreeClassifier ==========
acc : [0.95833333 0.875      0.91666667 0.95833333 0.95833333] avg acc : 0.9333
cross_val_predict : 0.9666666666666667
========== 아이리스 DummyClassifier ==========
acc : [0.33333333 0.33333333 0.33333333 0.33333333 0.33333333] avg acc : 0.3333
cross_val_predict : 0.3333333333333333
========== 아이리스 ExtraTreeClassifier ==========
acc : [0.95833333 0.875      0.91666667 0.91666667 0.95833333] avg acc : 0.925
cross_val_predict : 0.9333333333333333
========== 아이리스 ExtraTreesClassifier ==========
acc : [0.95833333 0.91666667 0.91666667 1.         0.95833333] avg acc : 0.95
cross_val_predict : 0.9666666666666667
FixedThresholdClassifier 는 예외처리
========== 아이리스 GaussianNB ==========
acc : [0.91666667 0.91666667 0.95833333 1.         0.95833333] avg acc : 0.95
cross_val_predict : 0.8666666666666667
========== 아이리스 GaussianProcessClassifier ==========
acc : [0.91666667 0.95833333 0.95833333 1.         0.91666667] avg acc : 0.95
cross_val_predict : 0.9666666666666667
========== 아이리스 GradientBoostingClassifier ==========
acc : [0.91666667 0.91666667 1.         0.95833333 0.95833333] avg acc : 0.95
cross_val_predict : 0.9
========== 아이리스 HistGradientBoostingClassifier ==========
acc : [0.95833333 0.91666667 0.95833333 0.95833333 0.95833333] avg acc : 0.95
cross_val_predict : 0.3333333333333333
========== 아이리스 KNeighborsClassifier ==========
acc : [0.91666667 1.         0.91666667 1.         1.        ] avg acc : 0.9667
cross_val_predict : 0.9333333333333333
========== 아이리스 LabelPropagation ==========
acc : [0.91666667 0.91666667 0.91666667 0.95833333 0.95833333] avg acc : 0.9333
cross_val_predict : 0.9
========== 아이리스 LabelSpreading ==========
acc : [0.91666667 0.91666667 0.91666667 0.95833333 0.95833333] avg acc : 0.9333
cross_val_predict : 0.9
========== 아이리스 LinearDiscriminantAnalysis ==========
acc : [0.95833333 0.91666667 1.         1.         1.        ] avg acc : 0.975
cross_val_predict : 1.0
========== 아이리스 LinearSVC ==========
acc : [0.95833333 0.91666667 0.95833333 0.95833333 0.95833333] avg acc : 0.95
cross_val_predict : 0.8666666666666667
========== 아이리스 LogisticRegression ==========
acc : [0.95833333 0.91666667 0.95833333 1.         1.        ] avg acc : 0.9667
cross_val_predict : 0.9
========== 아이리스 LogisticRegressionCV ==========
acc : [0.95833333 0.95833333 0.95833333 1.         1.        ] avg acc : 0.975
cross_val_predict : 0.9666666666666667
========== 아이리스 MLPClassifier ==========
acc : [0.95833333 0.91666667 0.95833333 1.         1.        ] avg acc : 0.9667
cross_val_predict : 0.9
MultiOutputClassifier 는 예외처리
MultinomialNB 는 예외처리
========== 아이리스 NearestCentroid ==========
acc : [0.91666667 0.83333333 0.875      0.95833333 0.83333333] avg acc : 0.8833
cross_val_predict : 0.7666666666666667
========== 아이리스 NuSVC ==========
acc : [0.91666667 0.95833333 0.95833333 1.         1.        ] avg acc : 0.9667
cross_val_predict : 0.9666666666666667
OneVsOneClassifier 는 예외처리
OneVsRestClassifier 는 예외처리
OutputCodeClassifier 는 예외처리
========== 아이리스 PassiveAggressiveClassifier ==========
acc : [0.95833333 0.875      0.95833333 0.875      0.91666667] avg acc : 0.9167
cross_val_predict : 0.8666666666666667
========== 아이리스 Perceptron ==========
acc : [0.79166667 0.66666667 0.95833333 0.79166667 0.79166667] avg acc : 0.8
cross_val_predict : 0.7666666666666667
========== 아이리스 QuadraticDiscriminantAnalysis ==========
acc : [0.95833333 0.91666667 1.         1.         1.        ] avg acc : 0.975
cross_val_predict : 0.9
========== 아이리스 RadiusNeighborsClassifier ==========
acc : [0.95833333        nan 0.95833333        nan        nan] avg acc : nan
RadiusNeighborsClassifier 는 예외처리
========== 아이리스 RandomForestClassifier ==========
acc : [0.95833333 0.91666667 0.95833333 0.95833333 0.91666667] avg acc : 0.9417
cross_val_predict : 0.9666666666666667
========== 아이리스 RidgeClassifier ==========
acc : [0.83333333 0.70833333 0.83333333 0.91666667 0.875     ] avg acc : 0.8333
cross_val_predict : 0.8666666666666667
========== 아이리스 RidgeClassifierCV ==========
acc : [0.83333333 0.70833333 0.79166667 0.91666667 0.875     ] avg acc : 0.825
cross_val_predict : 0.8333333333333334
========== 아이리스 SGDClassifier ==========
acc : [0.91666667 0.875      0.875      0.91666667 0.91666667] avg acc : 0.9
cross_val_predict : 0.8666666666666667
========== 아이리스 SVC ==========
acc : [0.95833333 0.95833333 0.95833333 1.         1.        ] avg acc : 0.975
cross_val_predict : 0.8666666666666667
StackingClassifier 는 예외처리
TunedThresholdClassifierCV 는 예외처리
VotingClassifier 는 예외처리
========== 캔서 AdaBoostClassifier ==========
acc : [0.96703297 0.97802198 0.94505495 0.91208791 0.96703297] avg acc : 0.9538
cross_val_predict : 0.9298245614035088
========== 캔서 BaggingClassifier ==========
acc : [0.93406593 0.95604396 0.93406593 0.9010989  0.96703297] avg acc : 0.9385
cross_val_predict : 0.9473684210526315
========== 캔서 BernoulliNB ==========
acc : [0.94505495 0.93406593 0.95604396 0.86813187 0.95604396] avg acc : 0.9319
cross_val_predict : 0.9473684210526315
========== 캔서 CalibratedClassifierCV ==========
acc : [0.97802198 1.         0.93406593 0.94505495 0.97802198] avg acc : 0.967
cross_val_predict : 0.9473684210526315
CategoricalNB 는 예외처리
ClassifierChain 는 예외처리
ComplementNB 는 예외처리
========== 캔서 DecisionTreeClassifier ==========
acc : [0.94505495 0.97802198 0.95604396 0.84615385 0.96703297] avg acc : 0.9385
cross_val_predict : 0.8771929824561403
========== 캔서 DummyClassifier ==========
acc : [0.62637363 0.62637363 0.62637363 0.62637363 0.62637363] avg acc : 0.6264
cross_val_predict : 0.631578947368421
========== 캔서 ExtraTreeClassifier ==========
acc : [0.94505495 0.93406593 0.91208791 0.92307692 0.92307692] avg acc : 0.9275
cross_val_predict : 0.8859649122807017
========== 캔서 ExtraTreesClassifier ==========
acc : [0.97802198 0.95604396 0.95604396 0.95604396 0.96703297] avg acc : 0.9626
cross_val_predict : 0.9385964912280702
FixedThresholdClassifier 는 예외처리
========== 캔서 GaussianNB ==========
acc : [0.95604396 0.96703297 0.9010989  0.92307692 0.92307692] avg acc : 0.9341
cross_val_predict : 0.956140350877193
========== 캔서 GaussianProcessClassifier ==========
acc : [0.98901099 0.97802198 0.94505495 0.96703297 0.97802198] avg acc : 0.9714
cross_val_predict : 0.9385964912280702
========== 캔서 GradientBoostingClassifier ==========
acc : [0.96703297 0.97802198 0.94505495 0.96703297 0.96703297] avg acc : 0.9648
cross_val_predict : 0.9122807017543859
========== 캔서 HistGradientBoostingClassifier ==========
acc : [1.         0.97802198 0.94505495 0.95604396 0.97802198] avg acc : 0.9714
cross_val_predict : 0.956140350877193
========== 캔서 KNeighborsClassifier ==========
acc : [0.97802198 0.97802198 0.97802198 0.95604396 0.95604396] avg acc : 0.9692
cross_val_predict : 0.956140350877193
========== 캔서 LabelPropagation ==========
acc : [0.98901099 0.95604396 0.93406593 0.92307692 0.95604396] avg acc : 0.9516
cross_val_predict : 0.9210526315789473
========== 캔서 LabelSpreading ==========
acc : [0.98901099 0.95604396 0.93406593 0.92307692 0.95604396] avg acc : 0.9516
cross_val_predict : 0.9210526315789473
========== 캔서 LinearDiscriminantAnalysis ==========
acc : [0.97802198 0.96703297 0.93406593 0.92307692 0.98901099] avg acc : 0.9582
cross_val_predict : 0.9385964912280702
========== 캔서 LinearSVC ==========
acc : [0.97802198 0.98901099 0.97802198 0.95604396 0.97802198] avg acc : 0.9758
cross_val_predict : 0.9473684210526315
========== 캔서 LogisticRegression ==========
acc : [1.         0.98901099 0.98901099 0.94505495 0.98901099] avg acc : 0.9824
cross_val_predict : 0.956140350877193
========== 캔서 LogisticRegressionCV ==========
acc : [0.98901099 0.97802198 0.98901099 0.95604396 0.97802198] avg acc : 0.978
cross_val_predict : 0.956140350877193
========== 캔서 MLPClassifier ==========
acc : [0.98901099 0.98901099 0.96703297 0.96703297 0.98901099] avg acc : 0.9802
cross_val_predict : 0.9385964912280702
MultiOutputClassifier 는 예외처리
MultinomialNB 는 예외처리
========== 캔서 NearestCentroid ==========
acc : [0.95604396 0.93406593 0.91208791 0.89010989 0.9010989 ] avg acc : 0.9187
cross_val_predict : 0.9385964912280702
========== 캔서 NuSVC ==========
acc : [0.97802198 0.95604396 0.92307692 0.89010989 0.94505495] avg acc : 0.9385
cross_val_predict : 0.9473684210526315
OneVsOneClassifier 는 예외처리
OneVsRestClassifier 는 예외처리
OutputCodeClassifier 는 예외처리
========== 캔서 PassiveAggressiveClassifier ==========
acc : [0.97802198 0.96703297 0.96703297 0.96703297 0.96703297] avg acc : 0.9692
cross_val_predict : 0.9473684210526315
========== 캔서 Perceptron ==========
acc : [0.96703297 0.97802198 0.98901099 0.94505495 0.97802198] avg acc : 0.9714
cross_val_predict : 0.9473684210526315
========== 캔서 QuadraticDiscriminantAnalysis ==========
acc : [0.93406593 0.97802198 0.94505495 0.93406593 0.95604396] avg acc : 0.9495
cross_val_predict : 0.7719298245614035
========== 캔서 RadiusNeighborsClassifier ==========
acc : [nan nan nan nan nan] avg acc : nan
RadiusNeighborsClassifier 는 예외처리
========== 캔서 RandomForestClassifier ==========
acc : [0.94505495 0.97802198 0.94505495 0.93406593 0.97802198] avg acc : 0.956
cross_val_predict : 0.9385964912280702
========== 캔서 RidgeClassifier ==========
acc : [0.97802198 0.97802198 0.92307692 0.92307692 0.98901099] avg acc : 0.9582
cross_val_predict : 0.9649122807017544
========== 캔서 RidgeClassifierCV ==========
acc : [0.97802198 0.97802198 0.92307692 0.93406593 0.98901099] avg acc : 0.9604
cross_val_predict : 0.9649122807017544
========== 캔서 SGDClassifier ==========
acc : [0.97802198 1.         0.98901099 0.95604396 0.98901099] avg acc : 0.9824
cross_val_predict : 0.9385964912280702
========== 캔서 SVC ==========
acc : [1.         0.98901099 0.97802198 0.94505495 0.97802198] avg acc : 0.978
cross_val_predict : 0.9649122807017544
StackingClassifier 는 예외처리
TunedThresholdClassifierCV 는 예외처리
VotingClassifier 는 예외처리
========== 와인 AdaBoostClassifier ==========
acc : [0.89655172 0.89655172 0.85714286 0.96428571 0.89285714] avg acc : 0.9015
cross_val_predict : 0.8611111111111112
========== 와인 BaggingClassifier ==========
acc : [0.93103448 0.89655172 1.         0.96428571 0.82142857] avg acc : 0.9227
cross_val_predict : 0.9444444444444444
========== 와인 BernoulliNB ==========
acc : [0.82758621 0.93103448 0.96428571 1.         0.85714286] avg acc : 0.916
cross_val_predict : 0.9444444444444444
========== 와인 CalibratedClassifierCV ==========
acc : [0.96551724 0.96551724 0.96428571 1.         1.        ] avg acc : 0.9791
cross_val_predict : 0.9722222222222222
CategoricalNB 는 예외처리
ClassifierChain 는 예외처리
ComplementNB 는 예외처리
========== 와인 DecisionTreeClassifier ==========
acc : [0.89655172 0.86206897 0.89285714 0.96428571 0.75      ] avg acc : 0.8732
cross_val_predict : 0.9166666666666666
========== 와인 DummyClassifier ==========
acc : [0.4137931  0.4137931  0.39285714 0.39285714 0.39285714] avg acc : 0.4012
cross_val_predict : 0.3888888888888889
========== 와인 ExtraTreeClassifier ==========
acc : [0.79310345 0.89655172 0.85714286 0.78571429 1.        ] avg acc : 0.8665
cross_val_predict : 0.75
========== 와인 ExtraTreesClassifier ==========
acc : [0.93103448 1.         1.         1.         1.        ] avg acc : 0.9862
cross_val_predict : 0.9166666666666666
FixedThresholdClassifier 는 예외처리
========== 와인 GaussianNB ==========
acc : [0.86206897 0.96551724 1.         1.         1.        ] avg acc : 0.9655
cross_val_predict : 0.9444444444444444
========== 와인 GaussianProcessClassifier ==========
acc : [0.93103448 0.96551724 1.         0.96428571 0.96428571] avg acc : 0.965
cross_val_predict : 0.9444444444444444
========== 와인 GradientBoostingClassifier ==========
acc : [0.82758621 0.89655172 0.96428571 0.92857143 0.89285714] avg acc : 0.902
cross_val_predict : 0.9166666666666666
========== 와인 HistGradientBoostingClassifier ==========
acc : [0.93103448 1.         1.         1.         1.        ] avg acc : 0.9862
cross_val_predict : 0.3888888888888889
========== 와인 KNeighborsClassifier ==========
acc : [0.89655172 0.96551724 1.         1.         0.96428571] avg acc : 0.9653
cross_val_predict : 0.9722222222222222
========== 와인 LabelPropagation ==========
acc : [0.89655172 0.96551724 1.         0.96428571 0.96428571] avg acc : 0.9581
cross_val_predict : 0.9166666666666666
========== 와인 LabelSpreading ==========
acc : [0.89655172 0.96551724 1.         0.96428571 0.96428571] avg acc : 0.9581
cross_val_predict : 0.9166666666666666
========== 와인 LinearDiscriminantAnalysis ==========
acc : [1.         0.96551724 1.         0.96428571 1.        ] avg acc : 0.986
cross_val_predict : 0.9444444444444444
========== 와인 LinearSVC ==========
acc : [1.         0.96551724 0.96428571 1.         1.        ] avg acc : 0.986
cross_val_predict : 0.9722222222222222
========== 와인 LogisticRegression ==========
acc : [0.96551724 1.         0.96428571 1.         1.        ] avg acc : 0.986
cross_val_predict : 0.9444444444444444
========== 와인 LogisticRegressionCV ==========
acc : [0.96551724 1.         1.         1.         0.96428571] avg acc : 0.986
cross_val_predict : 0.9722222222222222
========== 와인 MLPClassifier ==========
acc : [0.96551724 1.         1.         1.         1.        ] avg acc : 0.9931
cross_val_predict : 0.9722222222222222
MultiOutputClassifier 는 예외처리
MultinomialNB 는 예외처리
========== 와인 NearestCentroid ==========
acc : [0.89655172 0.96551724 1.         0.96428571 0.96428571] avg acc : 0.9581
cross_val_predict : 0.9722222222222222
========== 와인 NuSVC ==========
acc : [0.89655172 0.96551724 0.96428571 1.         1.        ] avg acc : 0.9653
cross_val_predict : 1.0
OneVsOneClassifier 는 예외처리
OneVsRestClassifier 는 예외처리
OutputCodeClassifier 는 예외처리
========== 와인 PassiveAggressiveClassifier ==========
acc : [1.         1.         0.96428571 1.         1.        ] avg acc : 0.9929
cross_val_predict : 0.9722222222222222
========== 와인 Perceptron ==========
acc : [0.96551724 0.96551724 0.96428571 0.96428571 1.        ] avg acc : 0.9719
cross_val_predict : 0.9722222222222222
========== 와인 QuadraticDiscriminantAnalysis ==========
acc : [0.96551724 0.96551724 0.92857143 1.         1.        ] avg acc : 0.9719
cross_val_predict : 0.5555555555555556
========== 와인 RadiusNeighborsClassifier ==========
acc : [nan nan nan nan nan] avg acc : nan
RadiusNeighborsClassifier 는 예외처리
========== 와인 RandomForestClassifier ==========
acc : [0.93103448 0.96551724 1.         1.         1.        ] avg acc : 0.9793
cross_val_predict : 0.9444444444444444
========== 와인 RidgeClassifier ==========
acc : [1.         1.         0.96428571 1.         1.        ] avg acc : 0.9929
cross_val_predict : 0.9722222222222222
========== 와인 RidgeClassifierCV ==========
acc : [1.         1.         0.96428571 1.         1.        ] avg acc : 0.9929
cross_val_predict : 0.9444444444444444
========== 와인 SGDClassifier ==========
acc : [1.         0.96551724 0.96428571 0.96428571 1.        ] avg acc : 0.9788
cross_val_predict : 0.9166666666666666
========== 와인 SVC ==========
acc : [0.96551724 0.96551724 0.96428571 1.         1.        ] avg acc : 0.9791
cross_val_predict : 1.0
StackingClassifier 는 예외처리
TunedThresholdClassifierCV 는 예외처리
VotingClassifier 는 예외처리
========== 디지트 AdaBoostClassifier ==========
acc : [0.28819444 0.36458333 0.25783972 0.26132404 0.24738676] avg acc : 0.2839
cross_val_predict : 0.2611111111111111
========== 디지트 BaggingClassifier ==========
acc : [0.94097222 0.92708333 0.91637631 0.95121951 0.91986063] avg acc : 0.9311
cross_val_predict : 0.8666666666666667
========== 디지트 BernoulliNB ==========
acc : [0.90625    0.88888889 0.89198606 0.8641115  0.87804878] avg acc : 0.8859
cross_val_predict : 0.85
========== 디지트 CalibratedClassifierCV ==========
acc : [0.96875    0.95833333 0.95121951 0.95470383 0.95818815] avg acc : 0.9582
cross_val_predict : 0.9305555555555556
CategoricalNB 는 예외처리
ClassifierChain 는 예외처리
ComplementNB 는 예외처리
========== 디지트 DecisionTreeClassifier ==========
acc : [0.82986111 0.82986111 0.86062718 0.80487805 0.81881533] avg acc : 0.8288
cross_val_predict : 0.7527777777777778
========== 디지트 DummyClassifier ==========
acc : [0.10069444 0.10069444 0.1010453  0.1010453  0.1010453 ] avg acc : 0.1009
cross_val_predict : 0.09722222222222222
========== 디지트 ExtraTreeClassifier ==========
acc : [0.78819444 0.73958333 0.71777003 0.73519164 0.75261324] avg acc : 0.7467
cross_val_predict : 0.6333333333333333
========== 디지트 ExtraTreesClassifier ==========
acc : [0.99305556 0.96875    0.97560976 0.97560976 0.98606272] avg acc : 0.9798
cross_val_predict : 0.9611111111111111
FixedThresholdClassifier 는 예외처리
========== 디지트 GaussianNB ==========
acc : [0.73958333 0.76388889 0.85714286 0.77700348 0.83275261] avg acc : 0.7941
cross_val_predict : 0.7916666666666666
========== 디지트 GaussianProcessClassifier ==========
acc : [0.96875    0.96875    0.97212544 0.95818815 0.96515679] avg acc : 0.9666
cross_val_predict : 0.9472222222222222
========== 디지트 GradientBoostingClassifier ==========
acc : [0.96180556 0.96875    0.95818815 0.96167247 0.95470383] avg acc : 0.961
cross_val_predict : 0.8805555555555555
========== 디지트 HistGradientBoostingClassifier ==========
acc : [0.97569444 0.95833333 0.97560976 0.97909408 0.96167247] avg acc : 0.9701
cross_val_predict : 0.9222222222222223
========== 디지트 KNeighborsClassifier ==========
acc : [0.97569444 0.96180556 0.97212544 0.95121951 0.96864111] avg acc : 0.9659
cross_val_predict : 0.9305555555555556
========== 디지트 LabelPropagation ==========
acc : [0.96180556 0.94791667 0.93031359 0.92334495 0.94076655] avg acc : 0.9408
cross_val_predict : 0.8861111111111111
========== 디지트 LabelSpreading ==========
acc : [0.96180556 0.94791667 0.93031359 0.92334495 0.94076655] avg acc : 0.9408
cross_val_predict : 0.8861111111111111
========== 디지트 LinearDiscriminantAnalysis ==========
acc : [0.96180556 0.94791667 0.94076655 0.94076655 0.93379791] avg acc : 0.945
cross_val_predict : 0.9277777777777778
========== 디지트 LinearSVC ==========
acc : [0.97222222 0.95833333 0.95121951 0.94773519 0.96167247] avg acc : 0.9582
cross_val_predict : 0.9277777777777778
========== 디지트 LogisticRegression ==========
acc : [0.97916667 0.95833333 0.95470383 0.95818815 0.96515679] avg acc : 0.9631
cross_val_predict : 0.95
========== 디지트 LogisticRegressionCV ==========
acc : [0.97569444 0.96180556 0.95470383 0.95470383 0.96515679] avg acc : 0.9624
cross_val_predict : 0.9444444444444444
========== 디지트 MLPClassifier ==========
acc : [0.99305556 0.96875    0.97212544 0.9825784  0.95818815] avg acc : 0.9749
cross_val_predict : 0.9527777777777777
MultiOutputClassifier 는 예외처리
MultinomialNB 는 예외처리
========== 디지트 NearestCentroid ==========
acc : [0.89583333 0.88541667 0.90592334 0.85714286 0.86759582] avg acc : 0.8824
cross_val_predict : 0.8888888888888888
========== 디지트 NuSVC ==========
acc : [0.95833333 0.94444444 0.95818815 0.90940767 0.94773519] avg acc : 0.9436
cross_val_predict : 0.9472222222222222
OneVsOneClassifier 는 예외처리
OneVsRestClassifier 는 예외처리
OutputCodeClassifier 는 예외처리
========== 디지트 PassiveAggressiveClassifier ==========
acc : [0.96180556 0.93402778 0.94425087 0.93031359 0.94076655] avg acc : 0.9422
cross_val_predict : 0.9222222222222223
========== 디지트 Perceptron ==========
acc : [0.94097222 0.92013889 0.93728223 0.95470383 0.94425087] avg acc : 0.9395
cross_val_predict : 0.8972222222222223
========== 디지트 QuadraticDiscriminantAnalysis ==========
acc : [0.86111111 0.83333333 0.8815331  0.88501742 0.88501742] avg acc : 0.8692
cross_val_predict : 0.3277777777777778
========== 디지트 RadiusNeighborsClassifier ==========
acc : [nan nan nan nan nan] avg acc : nan
RadiusNeighborsClassifier 는 예외처리
========== 디지트 RandomForestClassifier ==========
acc : [0.98263889 0.96875    0.96167247 0.97212544 0.97560976] avg acc : 0.9722
cross_val_predict : 0.9444444444444444
========== 디지트 RidgeClassifier ==========
acc : [0.92361111 0.91666667 0.94076655 0.91637631 0.90592334] avg acc : 0.9207
cross_val_predict : 0.9055555555555556
========== 디지트 RidgeClassifierCV ==========
acc : [0.93055556 0.92361111 0.94076655 0.91637631 0.90592334] avg acc : 0.9234
cross_val_predict : 0.9111111111111111
========== 디지트 SGDClassifier ==========
acc : [0.95833333 0.92708333 0.94773519 0.93728223 0.94425087] avg acc : 0.9429
cross_val_predict : 0.9277777777777778
========== 디지트 SVC ==========
acc : [0.98958333 0.97222222 0.9825784  0.97909408 0.9825784 ] avg acc : 0.9812
cross_val_predict : 0.9611111111111111
StackingClassifier 는 예외처리
TunedThresholdClassifierCV 는 예외처리
VotingClassifier 는 예외처리
'''
