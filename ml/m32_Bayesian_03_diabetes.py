import numpy as np
import time
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
import xgboost as xgb
from xgboost import XGBRegressor
from bayes_opt import BayesianOptimization
import warnings
warnings.filterwarnings('ignore')

# 1. 데이터
x, y = load_diabetes(return_X_y=True)

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
{'target': 0.5093814246195346, 'params': {'colsample_bytree': 0.5, 'learning_rate': 0.1, 'max_bin': 86.13552553831272, 'max_depth': 3.0, 'min_child_samples': 10.0, 'min_child_weight': 14.849861825025789, 'num_leaves': 24.0, 'reg_alpha': 30.831692310377456, 'reg_lambda': 10.0, 'subsample': 0.5}}
100 번 걸린시간 : 12.46 초

| 60        | 0.4979    | 0.5       | 0.1       | 85.81     | 3.0       | 10.0      | 14.0      | 24.0      | 30.39     | 10.0      | 0.5       |
| 61        | 0.4105    | 0.7225    | 0.01994   | 180.8     | 4.13      | 188.9     | 24.49     | 35.47     | 36.27     | 5.245     | 0.7729    |
| 62        | 0.4546    | 0.902     | 0.08266   | 175.3     | 5.254     | 189.6     | 29.35     | 32.94     | 32.32     | 4.752     | 0.6113    |
| 63        | 0.4553    | 0.7011    | 0.08004   | 488.3     | 7.799     | 37.32     | 48.54     | 31.22     | 7.688     | 4.944     | 0.6591    |
| 64        | 0.4784    | 0.8865    | 0.07289   | 170.6     | 8.199     | 190.0     | 24.66     | 29.36     | 28.1      | 4.347     | 0.5505    |
| 65        | 0.4192    | 0.6389    | 0.04587   | 77.31     | 6.33      | 12.21     | 9.366     | 29.33     | 44.57     | 9.319     | 0.9529    |
| 66        | 0.397     | 0.7493    | 0.09741   | 172.4     | 9.692     | 190.7     | 21.44     | 24.58     | 27.2      | 9.007     | 0.9257    |
| 67        | 0.3967    | 0.9782    | 0.06129   | 25.37     | 8.576     | 154.0     | 15.7      | 29.56     | 6.048     | 3.723     | 0.9525    |
| 68        | 0.4724    | 0.7108    | 0.07048   | 485.8     | 4.123     | 30.43     | 45.12     | 31.0      | 5.77      | 4.814     | 0.8807    |
| 69        | 0.1669    | 0.694     | 0.004435  | 177.9     | 3.632     | 187.3     | 29.77     | 29.79     | 29.95     | 5.958     | 0.9594    |
| 70        | 0.4268    | 0.7989    | 0.03265   | 169.8     | 5.738     | 185.9     | 20.61     | 31.89     | 28.27     | 0.1361    | 0.9114    |
| 71        | 0.4461    | 0.5339    | 0.03037   | 174.6     | 5.814     | 186.3     | 26.57     | 35.28     | 29.4      | 1.07      | 0.7116    |
| 72        | 0.4632    | 0.7329    | 0.04346   | 482.4     | 5.568     | 28.88     | 45.13     | 30.91     | 5.264     | 7.763     | 0.8762    |
| 73        | 0.4324    | 0.811     | 0.02611   | 180.2     | 6.132     | 188.9     | 31.31     | 39.71     | 34.3      | 7.14      | 0.8416    |
| 74        | 0.179     | 0.7193    | 0.004425  | 483.4     | 4.955     | 31.15     | 45.37     | 26.4      | 7.602     | 2.108     | 0.7967    |
| 75        | 0.5094    | 0.5       | 0.1       | 86.14     | 3.0       | 10.0      | 14.85     | 24.0      | 30.83     | 10.0      | 0.5       |
'''
