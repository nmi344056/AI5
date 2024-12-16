import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from sklearn.metrics import accuracy_score
from sklearn.utils import all_estimators

import sklearn as sk
import warnings         # warnings.warn( 없애기
warnings.filterwarnings('ignore')

#1. 데이터
x, y = fetch_california_housing(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, shuffle=True, random_state=123)

scaler = RobustScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델 구성
# all = all_estimators(type_filter='classifier')
all = all_estimators(type_filter='regressor')

# print('sk 버전 :', sk.__version__)    # 1.5.1
# print('all Algorithms :', all)
# print('모델의 갯수 :', len(all))      # 55 (sk 버전마다 다르다)

for name, model in all:
    try:
        #2. 모델
        model = model()
        #3. 훈련
        model.fit(x_train, y_train)     # 몇 번 되다 에러, 예외처리 필요, ValueError: `n_components` upper bound is 1. Got 2 instead. Reduce `n_components`.

        #4. 평가
        acc = model.score(x_test, y_test)
        print(name, '의 정답률 :', acc)
    except:
        print(name, '는 예외처리')

'''
ARDRegression 의 정답률 : 0.6100186362674526
AdaBoostRegressor 의 정답률 : 0.42560301048679605
BaggingRegressor 의 정답률 : 0.7913471980712417
BayesianRidge 의 정답률 : 0.6104356531406256
CCA 는 예외처리
DecisionTreeRegressor 의 정답률 : 0.593394718756097
DummyRegressor 의 정답률 : -2.3968123705309097e-05
ElasticNet 의 정답률 : 0.14343505618346586
ElasticNetCV 의 정답률 : 0.6097799853352929
ExtraTreeRegressor 의 정답률 : 0.5602186977190684
ExtraTreesRegressor 의 정답률 : 0.8187449616561343
GammaRegressor 의 정답률 : 0.2991204929739124
GaussianProcessRegressor 의 정답률 : -115.39820507988847
GradientBoostingRegressor 의 정답률 : 0.7979134984538584
HistGradientBoostingRegressor 의 정답률 : 0.8386513367548366
HuberRegressor 의 정답률 : 0.29813791530534217
IsotonicRegression 는 예외처리
KNeighborsRegressor 의 정답률 : 0.6903013931656454
KernelRidge 의 정답률 : -1.1255359965906533
Lars 의 정답률 : 0.6104546894797876
LarsCV 의 정답률 : 0.6089152295776292
Lasso 의 정답률 : -2.3968123705309097e-05
LassoCV 의 정답률 : 0.6102651711608287
LassoLars 의 정답률 : -2.3968123705309097e-05
LassoLarsCV 의 정답률 : 0.6104546894797876
LassoLarsIC 의 정답률 : 0.6104546894797876
LinearRegression 의 정답률 : 0.6104546894797875
LinearSVR 의 정답률 : -1.7242375465540438
MLPRegressor 의 정답률 : 0.7531970857947026
MultiOutputRegressor 는 예외처리
MultiTaskElasticNet 는 예외처리
MultiTaskElasticNetCV 는 예외처리
MultiTaskLasso 는 예외처리
MultiTaskLassoCV 는 예외처리
NuSVR 의 정답률 : 0.6917936987550335
OrthogonalMatchingPursuit 의 정답률 : 0.46885682123825645
OrthogonalMatchingPursuitCV 의 정답률 : 0.46885682123825645
PLSCanonical 는 예외처리
PLSRegression 의 정답률 : 0.5285950305889688
PassiveAggressiveRegressor 의 정답률 : -21.205654405812073
PoissonRegressor 의 정답률 : 0.4052896953891555
QuantileRegressor 의 정답률 : -0.05085046448429664
RANSACRegressor 의 정답률 : -2.9810768803646033
RadiusNeighborsRegressor 는 예외처리
RandomForestRegressor 의 정답률 : 0.8112357718482869
RegressorChain 는 예외처리
Ridge 의 정답률 : 0.6104269742873517
RidgeCV 의 정답률 : 0.6104269742878607
SGDRegressor 의 정답률 : -2.2020295951636367e+24
SVR 의 정답률 : 0.6874867483037101
StackingRegressor 는 예외처리
TheilSenRegressor 의 정답률 : -8.092428916137681
TransformedTargetRegressor 의 정답률 : 0.6104546894797875
TweedieRegressor 의 정답률 : 0.3416703812168921
VotingRegressor 는 예외처리
'''
