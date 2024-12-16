# https://www.kaggle.com/competitions/bike-sharing-demand/data?select=train.csv

import numpy as np
import pandas as pd
import time
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, GridSearchCV
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler
import xgboost as xgb
from xgboost import XGBRegressor
from bayes_opt import BayesianOptimization
import warnings
warnings.filterwarnings('ignore')

#1. 데이터
path = "C:\\ai5\\_data\\kaggle\\bike-sharing-demand\\"
# path = "C://ai5//_data//bike-sharing-demand//"
# path = "C://ai5/_data/bike-sharing-demand/"

train_csv = pd.read_csv(path + "train.csv", index_col=0)
test_csv = pd.read_csv(path + "test.csv", index_col=0)
sampleSubmission = pd.read_csv(path + "sampleSubmission.csv", index_col=0)

# print(train_csv.shape)              # (10886, 11)
# print(test_csv.shape)               # (6493, 8)
# print(sampleSubmission.shape)       # (6493, 1)

# print(train_csv.columns)
# Index(['season', 'holiday', 'workingday', 'weather', 'temp', 'atemp',
#        'humidity', 'windspeed', 'casual', 'registered', 'count'],    
#       dtype='object')

# print(train_csv.info())             # 결측치가 없다
# print(test_csv.info())              # 결측치가 없다
# print(train_csv.describe())

########## 결측치 확인 ##########
# print(train_csv.isna().sum())       # 0
# print(train_csv.isnull().sum())     # 0
# print(test_csv.isna().sum())        # 0
# print(test_csv.isnull().sum())      # 0

########## x와 y를 분리 ##########
x = train_csv.drop(['casual', 'registered', 'count'], axis=1)
# print(x)                            # [10886 rows x 8 columns]
y = train_csv['count']
# print(y)
# print(y.shape)                      # (10886,)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, random_state=1223)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

early_stop = xgb.callback.EarlyStopping(
    rounds=50,                      # patience
    # metric_name='logloss',        # 이진: logloss, 다중: mlogloss
    data_name='validation_0',
    # save_best=True,               # AttributeError: `best_iteration` is only defined when early stopping is used.
)

#2. 모델 구성
bayesian_params={
    'learning_rate' : (0.001, 0.1),
    'max_depth' : (3, 10),
    'num_leaves' : (24, 40),
    'min_child_samples' : (10, 200),
    'min_child_weight' : (1, 50),
    'subsample' : (0.5, 1),
    'colsample_bytree' : (0.5, 1),
    'max_bin' : (9, 500),
    'reg_lambda' : (-0.001, 10),
    'reg_alpha' : (0.01, 50),
}

def xgb_hamsu(learning_rate, max_depth, 
              num_leaves, min_child_samples, min_child_weight, 
              subsample, colsample_bytree, max_bin, reg_lambda, reg_alpha,
              ):
    params = {
        'n_estimators' : 100,
        'learning_rate' : learning_rate,
        'max_depth' : int(round(max_depth)),        # 무조건 정수형
        'num_leaves' : int(round(num_leaves)),
        'min_child_samples' : int(round(min_child_samples)),
        'min_child_weight' : int(round(min_child_weight)),
        'subsample' : max(min(subsample, 1), 0),    # 0 ~ 1 사이의 값
        'colsample_bytree' : colsample_bytree,
        'max_bin' : max(int(round(max_bin)), 10),   # 무조건 10 이상
        'reg_lambda' : max(reg_lambda, 0),          # 무조건 양수만
        'reg_alpha' : reg_alpha,
    }

    model = XGBRegressor(**params,
                        #  n_jobs=-1,
                         tree_mothod='gpu_hist',
                         gpu_id=0,
                         callbacks=[early_stop],
                         )

    model.fit(x_train, y_train,
            eval_set=[(x_test, y_test)],
            # eval_metric='logloss',
            verbose=0,
            )

    y_predict = model.predict(x_test)
    results = r2_score(y_test, y_predict)
    return results

bay = BayesianOptimization(                 # 모델 정의
    f=xgb_hamsu,
    pbounds=bayesian_params,
    random_state=333,
)

n_iter = 100
start = time.time()
bay.maximize(init_points=5, n_iter=n_iter)  # fit
end = time.time()

print(bay.max)
print(n_iter, '번 걸린시간 :', round(end - start, 2), '초')

'''
{'target': 0.3655242323875427, 'params': {'colsample_bytree': 1.0, 'learning_rate': 0.03829738841990449, 'max_bin': 70.88400079639472, 'max_depth': 10.0, 'min_child_samples': 20.40631492429576, 'min_child_weight': 
31.365385820740865, 'num_leaves': 40.0, 'reg_alpha': 50.0, 'reg_lambda': 10.0, 'subsample': 1.0}}
100 번 걸린시간 : 32.42 초

| 62        | 0.334     | 0.5       | 0.1       | 38.93     | 3.0       | 118.1     | 39.78     | 24.0      | 0.01      | -0.001    | 1.0       |
| 63        | 0.3347    | 0.6613    | 0.1       | 472.6     | 3.0       | 105.2     | 50.0      | 40.0      | 0.2326    | -0.001    | 1.0       |
| 64        | 0.3655    | 1.0       | 0.0383    | 70.88     | 10.0      | 20.41     | 31.37     | 40.0      | 50.0      | 10.0      | 1.0       |
'''
