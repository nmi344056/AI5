import numpy as np
import pandas as pd
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler
import xgboost as xgb
from xgboost import XGBRegressor
from bayes_opt import BayesianOptimization
import warnings
warnings.filterwarnings('ignore')

#1. 데이터
path = "./_data/dacon/따릉이/"

train_csv = pd.read_csv(path + "train.csv", index_col=0)
print(train_csv)            # [1459 rows x 11 columns] / [1459 rows x 10 columns]

test_csv = pd.read_csv(path + "test.csv", index_col=0)
print(test_csv)             # [715 rows x 9 columns]

submission_csv = pd.read_csv(path + "submission.csv", index_col=0)
print(submission_csv)       # [715 rows x 1 columns]

print(train_csv.columns)
# Index(['hour', 'hour_bef_temperature', 'hour_bef_precipitation',
#        'hour_bef_windspeed', 'hour_bef_humidity', 'hour_bef_visibility',
#        'hour_bef_ozone', 'hour_bef_pm10', 'hour_bef_pm2.5', 'count'],
#       dtype='object')

train_csv = train_csv.dropna()
test_csv = test_csv.fillna(test_csv.mean())

x = train_csv.drop(['count'], axis=1)
print(x)                    # [1328 rows x 9 columns]
y = train_csv['count']
print(y)
print(y.shape)              # (1328,)


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
{'target': 0.8135354050260377, 'params': {'colsample_bytree': 0.8554012931366352, 'learning_rate': 0.1, 'max_bin': 24.95168299519296, 'max_depth': 10.0, 'min_child_samples': 129.10484777160093, 'min_child_weight': 
13.790798471992447, 'num_leaves': 26.659084799016398, 'reg_alpha': 28.210441785493117, 'reg_lambda': 10.0, 'subsample': 0.890407147037931}}
100 번 걸린시간 : 38.14 초

| 6         | 0.7964    | 0.6555    | 0.09039   | 74.4      | 6.367     | 14.95     | 10.8      | 30.09     | 45.86     | 7.504     | 0.5537    |
| 7         | 0.8026    | 0.8415    | 0.05319   | 75.42     | 9.173     | 16.93     | 8.9       | 33.57     | 47.57     | 6.406     | 0.9704    |
| 8         | 0.7639    | 1.0       | 0.0617    | 20.88     | 10.0      | 12.84     | 1.0       | 40.0      | 41.5      | 10.0      | 1.0       |
| 9         | 0.7072    | 0.5       | 0.1       | 9.0       | 10.0      | 76.65     | 50.0      | 40.0      | 50.0      | 10.0      | 0.5       |
| 10        | 0.7664    | 1.0       | 0.1       | 9.0       | 3.696     | 146.6     | 1.0       | 24.0      | 0.01      | -0.001    | 1.0       |
| 11        | 0.3664    | 0.5851    | 0.005202  | 491.5     | 7.208     | 149.6     | 49.23     | 29.45     | 48.17     | 0.919     | 0.8014    |
| 12        | 0.8113    | 0.9539    | 0.1       | 398.3     | 10.0      | 200.0     | 1.0       | 40.0      | 33.11     | 10.0      | 0.94      |
| 13        | 0.7864    | 0.6557    | 0.06634   | 332.8     | 6.509     | 200.0     | 1.0       | 40.0      | 50.0      | -0.001    | 0.5       |
| 14        | 0.7757    | 1.0       | 0.1       | 355.5     | 10.0      | 200.0     | 44.47     | 24.0      | 0.01      | 10.0      | 1.0       |
| 15        | 0.7644    | 0.7509    | 0.1       | 363.7     | 10.0      | 149.0     | 1.0       | 40.0      | 0.03652   | -0.001    | 0.5       |

| 54        | 0.08229   | 0.5       | 0.001     | 56.73     | 3.0       | 115.8     | 1.0       | 36.64     | 0.01      | -0.001    | 0.5       |
| 55        | 0.8135    | 0.8554    | 0.1       | 24.95     | 10.0      | 129.1     | 13.79     | 26.66     | 28.21     | 10.0      | 0.8904    |
'''
