import numpy as np
import pandas as pd
import time
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import MinMaxScaler
import xgboost as xgb
from xgboost import XGBClassifier
from bayes_opt import BayesianOptimization
import warnings
warnings.filterwarnings('ignore')

#1. 데이터
x, y = load_digits(return_X_y=True)     #로 분할 가능
print(x)                    # [[]...[]]
print(y)                    # [0 1 2 ... 8 9 8]
print(x.shape, y.shape)     # (1797, 64) (1797,)

print(pd.value_counts(y, sort=False))
# 0    178
# 1    182
# 2    177
# 3    183
# 4    181
# 5    182
# 6    181
# 7    179
# 8    174
# 9    180

print(np.unique(y, return_counts=True))
# (array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]), array([178, 182, 177, 183, 181, 182, 181, 179, 174, 180], dtype=int64))

y_ohe1 = to_categorical(y)
print(y_ohe1)
print(y_ohe1.shape)         # (1797, 10)

x_train, x_test, y_train, y_test = train_test_split(x, y_ohe1, train_size=0.9, random_state=6666)

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
{'target': 0.9166666666666666, 'params': {'colsample_bytree': 0.5, 'learning_rate': 0.1, 'max_bin': 196.55342599449008, 'max_depth': 10.0, 'min_child_samples': 165.3398267598514, 'min_child_weight': 1.0, 'num_leaves': 37.518039474424924, 'reg_alpha': 0.01, 'reg_lambda': -0.001, 'subsample': 0.5364071060357155}}
100 번 걸린시간 : 50.22 초

| 9         | 0.8944    | 0.5034    | 0.1       | 9.0       | 10.0      | 42.32     | 1.0       | 40.0      | 0.01      | 10.0      | 0.6337    |
| 10        | 0.8722    | 0.5       | 0.1       | 79.13     | 10.0      | 200.0     | 1.0       | 24.0      | 0.01      | 10.0      | 0.5       |
| 11        | 0.8889    | 1.0       | 0.1       | 157.5     | 10.0      | 200.0     | 1.0       | 24.0      | 0.01      | -0.001    | 0.5       |
| 12        | 0.0       | 1.0       | 0.1       | 142.8     | 3.0       | 200.0     | 50.0      | 24.0      | 50.0      | 10.0      | 0.5       |
| 13        | 0.8722    | 0.5       | 0.1       | 66.72     | 10.0      | 142.3     | 1.0       | 24.0      | 0.01      | 10.0      | 0.5       |
| 14        | 0.8944    | 0.9976    | 0.1       | 239.0     | 10.0      | 200.0     | 1.0       | 24.0      | 0.01      | -0.001    | 0.5       |
| 15        | 0.0       | 1.0       | 0.001     | 322.1     | 3.0       | 200.0     | 1.0       | 40.0      | 0.01      | 10.0      | 1.0       |
| 16        | 0.9       | 0.5       | 0.1       | 201.5     | 10.0      | 152.9     | 1.0       | 24.0      | 0.01      | -0.001    | 0.5       |
| 17        | 0.0       | 1.0       | 0.1       | 500.0     | 10.0      | 10.0      | 50.0      | 40.0      | 0.01      | 10.0      | 0.5       |
| 18        | 0.5944    | 0.5       | 0.1       | 221.4     | 10.0      | 189.2     | 50.0      | 24.0      | 0.01      | -0.001    | 1.0       |
| 19        | 0.0       | 1.0       | 0.1       | 9.0       | 10.0      | 10.0      | 50.0      | 24.0      | 0.01      | 10.0      | 0.5       |

| 34        | 0.7611    | 0.5       | 0.1       | 61.34     | 10.0      | 126.0     | 14.69     | 24.0      | 0.01      | 9.133     | 0.5       |
| 35        | 0.0       | 1.0       | 0.001     | 227.5     | 3.0       | 193.8     | 1.0       | 38.33     | 0.01      | 10.0      | 0.5       |
| 36        | 0.9167    | 0.5       | 0.1       | 196.6     | 10.0      | 165.3     | 1.0       | 37.52     | 0.01      | -0.001    | 0.5364    |
'''
