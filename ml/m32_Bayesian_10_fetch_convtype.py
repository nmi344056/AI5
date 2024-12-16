import numpy as np
import pandas as pd
import time
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler
import xgboost as xgb
from xgboost import XGBRegressor
from bayes_opt import BayesianOptimization
import warnings
warnings.filterwarnings('ignore')

#1. 데이터
datasets = fetch_covtype()
x = datasets.data
y = datasets.target
print(x.shape, y.shape)     # (581012, 54) (581012,)

print(pd.value_counts(y))    
# 2    283301
# 1    211840
# 3     35754
# 7     20510
# 6     17367
# 5      9493
# 4      2747
# 문제 : 0이 없다, onehot을 0이 아닌 1부터 시작한다.

print(y)
print(np.unique(y, return_counts=True))

# from tensorflow.keras.utils import to_categorical
# y_ohe = to_categorical(y)
# print(y_ohe)
# print(y_ohe.shape)              # (581012, 8)

y_ohe = pd.get_dummies(y)          # pandas
print(y_ohe)                       # 1  2  3  4  5  6  7
print(y_ohe.shape)                 # (581012, 7)

# print("==============================")
# from sklearn.preprocessing import OneHotEncoder
# y_ohe = y.reshape(-1, 1)
# ohe = OneHotEncoder(sparse=False)        # True가 default
# ohe.fit(y_ohe)
# y_ohe = ohe.transform(y_ohe)
# print(y_ohe)
# print(y_ohe.shape)                 # (581012, 7)

x_train, x_test, y_train, y_test = train_test_split(x, y_ohe, train_size=0.9, random_state=6666,
                                                    stratify=y)

# print(pd.value_counts(y_train))
# 2    141429       141651
# 1    106155       105920
# 3     17958       17877
# 7     10262       10255
# 6      8613       8683
# 5      4711       4747
# 4      1378       1373

print(x_train.shape, x_test.shape)      # (522910, 54) (58102, 54)
print(y_train.shape, y_test.shape)      # (522910, 8) (58102, 8)

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
{'target': 0.7905117273330688, 'params': {'colsample_bytree': 1.0, 'learning_rate': 0.1, 'max_bin': 470.3783050537351, 'max_depth': 10.0, 'min_child_samples': 200.0, 'min_child_weight': 2.146633275321259, 'num_leaves': 24.0, 'reg_alpha': 10.09953245098034, 'reg_lambda': 10.0, 'subsample': 1.0}}
100 번 걸린시간 : 1428.56 초

| 30        | 0.7633    | 1.0       | 0.1       | 473.7     | 10.0      | 200.0     | 8.417     | 24.0      | 19.92     | 10.0      | 1.0       |
| 31        | 0.7546    | 1.0       | 0.1       | 470.5     | 10.0      | 198.1     | 9.41      | 26.02     | 24.18     | 2.135     | 1.0       |
| 32        | 0.692     | 1.0       | 0.04075   | 478.0     | 10.0      | 193.5     | 12.09     | 28.85     | 21.79     | 6.849     | 1.0       |
| 33        | 0.7722    | 1.0       | 0.1       | 471.3     | 10.0      | 192.6     | 5.843     | 24.0      | 16.77     | 4.845     | 1.0       |
| 34        | 0.7656    | 1.0       | 0.1       | 475.3     | 10.0      | 199.1     | 1.0       | 27.1      | 20.26     | 4.137     | 1.0       |
| 35        | 0.7515    | 1.0       | 0.1       | 478.4     | 10.0      | 200.0     | 1.869     | 33.06     | 26.39     | 10.0      | 1.0       |
| 36        | 0.7797    | 1.0       | 0.1       | 467.4     | 10.0      | 200.0     | 11.11     | 24.0      | 13.75     | 4.046     | 1.0       |
| 37        | 0.7852    | 1.0       | 0.1       | 473.7     | 10.0      | 200.0     | 6.815     | 30.75     | 11.86     | 4.926     | 1.0       |
| 38        | 0.7904    | 1.0       | 0.1       | 470.4     | 10.0      | 200.0     | 1.978     | 24.0      | 10.9      | -0.001    | 1.0       |
| 39        | 0.7905    | 1.0       | 0.1       | 470.4     | 10.0      | 200.0     | 2.147     | 24.0      | 10.1      | 10.0      | 1.0       |
'''
