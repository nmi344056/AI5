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
ARDRegression 는 예외처리
AdaBoostRegressor 는 예외처리
BaggingRegressor 는 예외처리
BayesianRidge 는 예외처리
CCA 는 예외처리
DecisionTreeRegressor 는 예외처리
DummyRegressor 는 예외처리
ElasticNet 는 예외처리
ElasticNetCV 는 예외처리
ExtraTreeRegressor 는 예외처리
ExtraTreesRegressor 는 예외처리
GammaRegressor 는 예외처리
GaussianProcessRegressor 는 예외처리
GradientBoostingRegressor 는 예외처리
HistGradientBoostingRegressor 는 예외처리
HuberRegressor 는 예외처리
IsotonicRegression 는 예외처리
KNeighborsRegressor 는 예외처리
KernelRidge 는 예외처리
Lars 는 예외처리
LarsCV 는 예외처리
Lasso 는 예외처리
LassoCV 는 예외처리
LassoLars 는 예외처리
LassoLarsCV 는 예외처리
LassoLarsIC 는 예외처리
LinearRegression 는 예외처리
LinearSVR 는 예외처리
MLPRegressor 는 예외처리
MultiOutputRegressor 는 예외처리
MultiTaskElasticNet 는 예외처리
MultiTaskElasticNetCV 는 예외처리
MultiTaskLasso 는 예외처리
MultiTaskLassoCV 는 예외처리
NuSVR 는 예외처리
OrthogonalMatchingPursuit 는 예외처리
OrthogonalMatchingPursuitCV 는 예외처리
PLSCanonical 는 예외처리
PLSRegression 는 예외처리
PassiveAggressiveRegressor 는 예외처리
PoissonRegressor 는 예외처리
RANSACRegressor 는 예외처리
RadiusNeighborsRegressor 는 예외처리
RandomForestRegressor 는 예외처리
RegressorChain 는 예외처리
Ridge 는 예외처리
RidgeCV 는 예외처리
SGDRegressor 는 예외처리
SVR 는 예외처리
StackingRegressor 는 예외처리
TheilSenRegressor 는 예외처리
TransformedTargetRegressor 는 예외처리
TweedieRegressor 는 예외처리
VotingRegressor 는 예외처리
ARDRegression 는 예외처리
AdaBoostRegressor 는 예외처리
BaggingRegressor 는 예외처리
BayesianRidge 는 예외처리
CCA 는 예외처리
DecisionTreeRegressor 는 예외처리
DummyRegressor 는 예외처리
ElasticNet 는 예외처리
ElasticNetCV 는 예외처리
ExtraTreeRegressor 는 예외처리
ExtraTreesRegressor 는 예외처리
GammaRegressor 는 예외처리
GaussianProcessRegressor 는 예외처리
GradientBoostingRegressor 는 예외처리
HistGradientBoostingRegressor 는 예외처리
HuberRegressor 는 예외처리
IsotonicRegression 는 예외처리
KNeighborsRegressor 는 예외처리
KernelRidge 는 예외처리
Lars 는 예외처리
LarsCV 는 예외처리
Lasso 는 예외처리
LassoCV 는 예외처리
LassoLars 는 예외처리
LassoLarsCV 는 예외처리
LassoLarsIC 는 예외처리
LinearRegression 는 예외처리
LinearSVR 는 예외처리
MLPRegressor 는 예외처리
MultiOutputRegressor 는 예외처리
MultiTaskElasticNet 는 예외처리
MultiTaskElasticNetCV 는 예외처리
MultiTaskLasso 는 예외처리
MultiTaskLassoCV 는 예외처리
NuSVR 는 예외처리
OrthogonalMatchingPursuit 는 예외처리
OrthogonalMatchingPursuitCV 는 예외처리
PLSCanonical 는 예외처리
PLSRegression 는 예외처리
PassiveAggressiveRegressor 는 예외처리
PoissonRegressor 는 예외처리
RANSACRegressor 는 예외처리
RadiusNeighborsRegressor 는 예외처리
RandomForestRegressor 는 예외처리
RegressorChain 는 예외처리
Ridge 는 예외처리
RidgeCV 는 예외처리
SGDRegressor 는 예외처리
SVR 는 예외처리
StackingRegressor 는 예외처리
TheilSenRegressor 는 예외처리
TransformedTargetRegressor 는 예외처리
TweedieRegressor 는 예외처리
VotingRegressor 는 예외처리
========== 당뇨병 ARDRegression ==========
acc : [0.33669511 0.51614173 0.45714001 0.4942447  0.49810389] avg acc : 0.4605
ARDRegression 는 예외처리
========== 당뇨병 AdaBoostRegressor ==========
acc : [0.27491923 0.44194024 0.35795628 0.42754247 0.43253195] avg acc : 0.387
AdaBoostRegressor 는 예외처리
========== 당뇨병 BaggingRegressor ==========
acc : [0.30247285 0.43910334 0.3695176  0.42226548 0.31946812] avg acc : 0.3706
BaggingRegressor 는 예외처리
========== 당뇨병 BayesianRidge ==========
acc : [0.33665099 0.51845177 0.46953911 0.49999988 0.5129947 ] avg acc : 0.4675
BayesianRidge 는 예외처리
========== 당뇨병 CCA ==========
acc : [0.17156797 0.48490022 0.26531834 0.48554646 0.51545541] avg acc : 0.3846
CCA 는 예외처리
========== 당뇨병 DecisionTreeRegressor ==========
acc : [ 0.13531062 -0.06481866 -0.23839774 -0.06496886  0.16941728] avg acc : -0.0127
DecisionTreeRegressor 는 예외처리
========== 당뇨병 DummyRegressor ==========
acc : [-0.03169715 -0.00054737 -0.00032752 -0.02427878 -0.00023281] avg acc : -0.0114
DummyRegressor 는 예외처리
========== 당뇨병 ElasticNet ==========
acc : [0.36498448 0.50632987 0.45433105 0.46037821 0.47139137] avg acc : 0.4515
ElasticNet 는 예외처리
========== 당뇨병 ElasticNetCV ==========
acc : [0.34091254 0.51772177 0.46941895 0.50039981 0.5142161 ] avg acc : 0.4685
ElasticNetCV 는 예외처리
========== 당뇨병 ExtraTreeRegressor ==========
acc : [-0.00281335 -0.28387712 -0.58947821 -0.16495086 -0.22142552] avg acc : -0.2525
ExtraTreeRegressor 는 예외처리
========== 당뇨병 ExtraTreesRegressor ==========
acc : [0.32851231 0.49915131 0.41006577 0.5020442  0.45830195] avg acc : 0.4396
ExtraTreesRegressor 는 예외처리
========== 당뇨병 GammaRegressor ==========
acc : [0.33818478 0.4230219  0.36267094 0.36526029 0.39998695] avg acc : 0.3778
GammaRegressor 는 예외처리
========== 당뇨병 GaussianProcessRegressor ==========
acc : [-0.31825292 -0.99329767 -0.93078663 -0.78789975 -0.97381734] avg acc : -0.8008
GaussianProcessRegressor 는 예외처리
========== 당뇨병 GradientBoostingRegressor ==========
acc : [0.28418246 0.43092502 0.29744592 0.48279675 0.4025512 ] avg acc : 0.3796
GradientBoostingRegressor 는 예외처리
========== 당뇨병 HistGradientBoostingRegressor ==========
acc : [0.31109447 0.42113154 0.32226817 0.39178551 0.33696915] avg acc : 0.3566
HistGradientBoostingRegressor 는 예외처리
========== 당뇨병 HuberRegressor ==========
acc : [0.32391391 0.48305237 0.48149987 0.50879297 0.51819314] avg acc : 0.4631
HuberRegressor 는 예외처리
========== 당뇨병 IsotonicRegression ==========
acc : [nan nan nan nan nan] avg acc : nan
IsotonicRegression 는 예외처리
========== 당뇨병 KNeighborsRegressor ==========
acc : [0.2516382  0.39699501 0.33406492 0.42779025 0.38546989] avg acc : 0.3592
KNeighborsRegressor 는 예외처리
========== 당뇨병 KernelRidge ==========
acc : [-3.62857319 -4.02643463 -4.79362132 -3.6569443  -3.29644923] avg acc : -3.8804
KernelRidge 는 예외처리
========== 당뇨병 Lars ==========
acc : [ 0.33497471 -0.65330145  0.49223519  0.50545055  0.51439104] avg acc : 0.2388
Lars 는 예외처리
========== 당뇨병 LarsCV ==========
acc : [0.34871384 0.52094943 0.46469784 0.50545055 0.48925832] avg acc : 0.4658
LarsCV 는 예외처리
========== 당뇨병 Lasso ==========
acc : [0.34530309 0.51774499 0.4693698  0.49429271 0.50891451] avg acc : 0.4671
Lasso 는 예외처리
========== 당뇨병 LassoCV ==========
acc : [0.33137888 0.49893358 0.4705489  0.50491086 0.5151866 ] avg acc : 0.4642
LassoCV 는 예외처리
========== 당뇨병 LassoLars ==========
acc : [0.29003094 0.36599321 0.39489411 0.3648589  0.38869289] avg acc : 0.3609
LassoLars 는 예외처리
========== 당뇨병 LassoLarsCV ==========
acc : [0.33163406 0.49479623 0.4704499  0.50519209 0.51439104] avg acc : 0.4633
LassoLarsCV 는 예외처리
========== 당뇨병 LassoLarsIC ==========
acc : [0.35914267 0.5117113  0.45897694 0.49645636 0.50426041] avg acc : 0.4661
LassoLarsIC 는 예외처리
========== 당뇨병 LinearRegression ==========
acc : [0.33497471 0.49479623 0.47751714 0.50545055 0.51439104] avg acc : 0.4654
LinearRegression 는 예외처리
========== 당뇨병 LinearSVR ==========
acc : [0.22295408 0.21089923 0.13003127 0.17865354 0.28205302] avg acc : 0.2049
LinearSVR 는 예외처리
========== 당뇨병 MLPRegressor ==========
acc : [-0.74378093 -0.96139455 -1.3842722  -1.056516   -1.08009768] avg acc : -1.0452
MLPRegressor 는 예외처리
MultiOutputRegressor 는 예외처리
========== 당뇨병 MultiTaskElasticNet ==========
acc : [nan nan nan nan nan] avg acc : nan
MultiTaskElasticNet 는 예외처리
========== 당뇨병 MultiTaskElasticNetCV ==========
acc : [nan nan nan nan nan] avg acc : nan
MultiTaskElasticNetCV 는 예외처리
========== 당뇨병 MultiTaskLasso ==========
acc : [nan nan nan nan nan] avg acc : nan
MultiTaskLasso 는 예외처리
========== 당뇨병 MultiTaskLassoCV ==========
acc : [nan nan nan nan nan] avg acc : nan
MultiTaskLassoCV 는 예외처리
========== 당뇨병 NuSVR ==========
acc : [0.09715658 0.14412681 0.13886826 0.10466764 0.1317235 ] avg acc : 0.1233
NuSVR 는 예외처리
========== 당뇨병 OrthogonalMatchingPursuit ==========
acc : [0.25826301 0.33730432 0.23419754 0.38511397 0.37314117] avg acc : 0.3176
OrthogonalMatchingPursuit 는 예외처리
========== 당뇨병 OrthogonalMatchingPursuitCV ==========
acc : [0.33335455 0.48590378 0.45304964 0.47233534 0.47194642] avg acc : 0.4433
OrthogonalMatchingPursuitCV 는 예외처리
========== 당뇨병 PLSCanonical ==========
acc : [-1.21707105 -0.81347009 -1.99005736 -1.21886149 -0.94928115] avg acc : -1.2377
PLSCanonical 는 예외처리
========== 당뇨병 PLSRegression ==========
acc : [0.36302341 0.51555493 0.45666779 0.48813789 0.50403139] avg acc : 0.4655
PLSRegression 는 예외처리
========== 당뇨병 PassiveAggressiveRegressor ==========
acc : [0.23399111 0.50752988 0.46937677 0.46476583 0.46519388] avg acc : 0.4282
PassiveAggressiveRegressor 는 예외처리
========== 당뇨병 PoissonRegressor ==========
acc : [0.37078092 0.51089455 0.43368963 0.47180183 0.50290496] avg acc : 0.458
PoissonRegressor 는 예외처리
========== 당뇨병 RANSACRegressor ==========
acc : [-0.1310029  -0.19135941  0.01504694  0.23744268  0.12876904] avg acc : 0.0118
RANSACRegressor 는 예외처리
========== 당뇨병 RadiusNeighborsRegressor ==========
acc : [nan nan nan nan nan] avg acc : nan
RadiusNeighborsRegressor 는 예외처리
========== 당뇨병 RandomForestRegressor ==========
acc : [0.32254473 0.48225196 0.38091269 0.46145858 0.49073868] avg acc : 0.4276
RandomForestRegressor 는 예외처리
RegressorChain 는 예외처리
========== 당뇨병 Ridge ==========
acc : [0.33255816 0.50295461 0.47453965 0.50448178 0.51598795] avg acc : 0.4661
Ridge 는 예외처리
========== 당뇨병 RidgeCV ==========
acc : [0.33442628 0.50295461 0.46965494 0.50136296 0.51598795] avg acc : 0.4649
RidgeCV 는 예외처리
========== 당뇨병 SGDRegressor ==========
acc : [0.32491888 0.51534892 0.46927892 0.50287763 0.51831684] avg acc : 0.4661
SGDRegressor 는 예외처리
========== 당뇨병 SVR ==========
acc : [0.12568127 0.12491259 0.09685269 0.06213539 0.12463563] avg acc : 0.1068
SVR 는 예외처리
StackingRegressor 는 예외처리
========== 당뇨병 TheilSenRegressor ==========
acc : [0.30229064 0.48749704 0.46673621 0.49030015 0.51422873] avg acc : 0.4522
TheilSenRegressor 는 예외처리
========== 당뇨병 TransformedTargetRegressor ==========
acc : [0.33497471 0.49479623 0.47751714 0.50545055 0.51439104] avg acc : 0.4654
TransformedTargetRegressor 는 예외처리
========== 당뇨병 TweedieRegressor ==========
acc : [0.35118826 0.47523401 0.4305841  0.42607965 0.43588961] avg acc : 0.4238
TweedieRegressor 는 예외처리
VotingRegressor 는 예외처리
'''
