import numpy as np
import time
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler
import xgboost as xgb
from xgboost import XGBRegressor
from bayes_opt import BayesianOptimization
import warnings
warnings.filterwarnings('ignore')

# 1. 데이터
x, y = fetch_california_housing(return_X_y=True)

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
{'target': 0.8518362039337846, 'params': {'colsample_bytree': 0.5, 'learning_rate': 0.1, 'max_bin': 56.56550436779346, 'max_depth': 8.81777771125184, 'min_child_samples': 92.37716195444311, 'min_child_weight': 7.303062416007875, 'num_leaves': 39.652041726795574, 'reg_alpha': 2.6210074657598095, 'reg_lambda': -0.001, 'subsample': 1.0}}
100 번 걸린시간 : 25.17 초

| 56        | 0.8471    | 0.5       | 0.1       | 57.04     | 9.468     | 91.58     | 6.457     | 39.66     | 3.551     | -0.001    | 0.8549    |
| 57        | 0.8454    | 0.7909    | 0.1       | 57.38     | 9.474     | 91.31     | 6.508     | 39.57     | 2.956     | -0.001    | 0.7412    |
| 58        | 0.8372    | 1.0       | 0.1       | 56.9      | 9.346     | 91.7      | 6.965     | 39.48     | 3.178     | -0.001    | 1.0       |
| 59        | 0.7329    | 0.6358    | 0.0292    | 490.8     | 4.827     | 27.93     | 35.1      | 26.94     | 39.45     | 7.238     | 0.8347    |
| 60        | 0.8442    | 0.5419    | 0.1       | 56.69     | 9.422     | 91.25     | 6.932     | 39.8      | 3.088     | -0.001    | 0.5       |
| 61        | 0.8506    | 0.5       | 0.1       | 57.0      | 8.707     | 91.42     | 6.704     | 39.8      | 3.039     | -0.001    | 0.9681    |
| 62        | 0.8446    | 0.7423    | 0.1       | 57.12     | 9.075     | 92.03     | 6.731     | 40.0      | 3.027     | -0.001    | 0.5       |
| 63        | 0.8451    | 0.5       | 0.1       | 57.03     | 8.962     | 91.8      | 6.819     | 39.07     | 2.925     | -0.001    | 0.5       |
| 64        | 0.8425    | 0.5       | 0.1       | 56.46     | 8.743     | 91.91     | 7.034     | 39.63     | 3.552     | -0.001    | 0.527     |
| 65        | 0.8471    | 0.5       | 0.1       | 56.64     | 9.665     | 92.35     | 7.168     | 39.44     | 3.239     | -0.001    | 0.5       |
| 66        | 0.8508    | 0.5       | 0.1       | 57.16     | 8.85      | 92.56     | 6.876     | 39.35     | 3.483     | -0.001    | 1.0       |
| 67        | 0.8469    | 0.5       | 0.1       | 57.25     | 8.937     | 91.99     | 7.706     | 39.61     | 3.194     | -0.001    | 0.6039    |
| 68        | 0.8518    | 0.5       | 0.1       | 56.57     | 8.818     | 92.38     | 7.303     | 39.65     | 2.621     | -0.001    | 1.0       |
'''
