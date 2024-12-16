# https://www.kaggle.com/competitions/otto-group-product-classification-challenge/overview

import numpy as np
import pandas as pd
import time
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler
import xgboost as xgb
from xgboost import XGBRegressor
from bayes_opt import BayesianOptimization
import warnings
warnings.filterwarnings('ignore')

#1. 데이터
path = "C:\\ai5\\_data\\kaggle\\otto-group-product-classification-challenge\\"

train_csv = pd.read_csv(path + "train.csv", index_col=0)
test_csv = pd.read_csv(path + "test.csv", index_col=0)
submission_csv = pd.read_csv(path + "sampleSubmission.csv", index_col=0)

print(train_csv.isna().sum())       # 결측치 없음
print(test_csv.isna().sum())        # 결측치 없음

print(train_csv)                    # target이 Class_1 ... Class_9

encoder = LabelEncoder()
train_csv['target'] = encoder.fit_transform(train_csv['target'])    # 라벨링

print(train_csv)                    # target이 0 ... 8

x = train_csv.drop(['target'], axis=1)
y = train_csv['target']

print(x.shape)                      # (61878, 93), =input_dim
print(y.shape)                      # (61878,)
print(y)
# 1        0
# 2        0
# 3        0
# 4        0
# 5        0
#         ..
# 61874    8
# 61875    8
# 61876    8
# 61877    8
# 61878    8

print(pd.value_counts(y, sort=False))   # 라벨 별 카운트
# 0     1929
# 1    16122
# 2     8004
# 3     2691
# 4     2739
# 5    14135
# 6     2839
# 7     8464
# 8     4955

y_ohe = pd.get_dummies(y)           # OneHot

print(y_ohe.shape)                  # (61878, 9) ,9의 추가를 통해 OneHot 학인, =output_dim
print(y_ohe)
# 1      1  0  0  0  0  0  0  0  0
# 2      1  0  0  0  0  0  0  0  0
# 3      1  0  0  0  0  0  0  0  0
# 4      1  0  0  0  0  0  0  0  0
# 5      1  0  0  0  0  0  0  0  0
# ...   .. .. .. .. .. .. .. .. ..
# 61874  0  0  0  0  0  0  0  0  1
# 61875  0  0  0  0  0  0  0  0  1
# 61876  0  0  0  0  0  0  0  0  1
# 61877  0  0  0  0  0  0  0  0  1
# 61878  0  0  0  0  0  0  0  0  1

print(pd.value_counts(y, sort=False))   # 동일하게 나온다, 컴퓨터가 인식할때는 숫자로 인식해서?

x_train, x_test, y_train, y_test = train_test_split(x, y_ohe, train_size=0.9, random_state=3, stratify=y)

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
{'target': 0.6378445029258728, 'params': {'colsample_bytree': 1.0, 'learning_rate': 0.1, 'max_bin': 180.36605965666544, 'max_depth': 10.0, 'min_child_samples': 200.0, 'min_child_weight': 3.0082824140488196, 'num_leaves': 24.0, 'reg_alpha': 0.01, 'reg_lambda': 10.0, 'subsample': 0.972912215366156}}
100 번 걸린시간 : 255.62 초

| 50        | 0.6326    | 1.0       | 0.1       | 85.38     | 10.0      | 108.1     | 15.72     | 24.0      | 0.01      | -0.001    | 1.0       |
| 51        | 0.587     | 1.0       | 0.1       | 82.87     | 10.0      | 113.4     | 9.191     | 24.0      | 36.93     | -0.001    | 1.0       |
| 52        | 0.5654    | 1.0       | 0.1       | 174.0     | 10.0      | 200.0     | 50.0      | 24.0      | 28.61     | 10.0      | 0.5       |
| 53        | 0.6351    | 1.0       | 0.1       | 170.1     | 10.0      | 167.3     | 1.0       | 24.0      | 0.01      | 10.0      | 0.5       |
| 54        | 0.6378    | 1.0       | 0.1       | 180.4     | 10.0      | 200.0     | 3.008     | 24.0      | 0.01      | 10.0      | 0.9729    |
'''
