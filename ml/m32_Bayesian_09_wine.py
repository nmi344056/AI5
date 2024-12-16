import numpy as np
import time
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.metrics import r2_score, accuracy_score
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler
import xgboost as xgb
from xgboost import XGBClassifier
from bayes_opt import BayesianOptimization
import warnings
warnings.filterwarnings('ignore')

#1. 데이터
x, y = load_wine(return_X_y=True)
# print(x)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, shuffle=True, random_state=123)

scaler = MaxAbsScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

early_stop = xgb.callback.EarlyStopping(
    rounds=50,                      # patience
    # metric_name='logloss',        # 이진: logloss, 다중: mlogloss
    data_name='validation_0',
    # save_best=True,               # AttributeError: `best_iteration` is only defined when early stopping is used.
)

# n_splits = 5
# # kfold = KFold(n_splits=n_splits, shuffle=True, random_state=123)
# kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=123)

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

    model = XGBClassifier(**params,
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
    results = accuracy_score(y_test, y_predict)
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
{'target': 1.0, 'params': {'colsample_bytree': 0.7716455435126173, 'learning_rate': 0.07316612206200168, 'max_bin': 17.28879000553614, 'max_depth': 5.312371620396405, 'min_child_samples': 80.05714672806884, 'min_child_weight': 3.3668796217250136, 'num_leaves': 25.672483010759535, 'reg_alpha': 4.880901405662105, 'reg_lambda': 2.453278548450057, 'subsample': 0.9367446772294055}}
100 번 걸린시간 : 43.72 초

|   iter    |  target   | colsam... | learni... |  max_bin  | max_depth | min_ch... | min_ch... | num_le... | reg_alpha | reg_la... | subsample |
-------------------------------------------------------------------------------------------------------------------------------------------------
| 1         | 1.0       | 0.7716    | 0.07317   | 17.29     | 5.312     | 80.06     | 3.367     | 25.67     | 4.881     | 2.453     | 0.9367    |
| 2         | 0.5278    | 0.5258    | 0.006988  | 126.2     | 4.426     | 43.16     | 21.35     | 39.07     | 23.35     | 9.6       | 0.9999    |
| 3         | 0.2222    | 0.8632    | 0.04894   | 71.29     | 3.738     | 18.03     | 12.96     | 25.14     | 49.78     | 2.012     | 0.9412    |
'''
