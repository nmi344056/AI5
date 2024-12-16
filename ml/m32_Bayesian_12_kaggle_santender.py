# https://www.kaggle.com/competitions/santander-customer-transaction-prediction

import numpy as np
import pandas as pd
import time
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
path = "C:/ai5/_data/kaggle/santander-customer-transaction-prediction/"

train_csv = pd.read_csv(path + "train.csv", index_col=0)
test_csv = pd.read_csv(path + "test.csv", index_col=0)
submission_csv = pd.read_csv(path + "sample_submission.csv", index_col=0)

print(train_csv.isna().sum())   # 결측치 없음
print(test_csv.isna().sum())   # 결측치 없음

x = train_csv.drop(['target'], axis=1)
y = train_csv['target']

print(x.shape)  # (200000, 200)
print(y.shape)  # (200000,)

print(pd.value_counts(y, sort=True))    # 이진 분류
# 0    179902
# 1     20098

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=5233,
                                                    stratify=y)

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
{'target': 0.9087, 'params': {'colsample_bytree': 0.5787426153323638, 'learning_rate': 0.1, 'max_bin': 397.8996059710389, 'max_depth': 6.992982033284695, 'min_child_samples': 49.9858090827484, 'min_child_weight': 30.679194622670064, 'num_leaves': 37.86255060315542, 'reg_alpha': 4.313335505037652, 'reg_lambda': 0.5276704113596514, 'subsample': 0.7845490597561333}}
100 번 걸린시간 : 477.17 초

| 63        | 0.9061    | 0.5257    | 0.09397   | 397.9     | 7.033     | 50.06     | 30.7      | 37.94     | 4.276     | 0.5562    | 0.9333    |
| 64        | 0.9065    | 0.5237    | 0.09227   | 397.9     | 7.031     | 50.06     | 30.7      | 37.93     | 4.274     | 0.5543    | 0.9313    |
| 65        | 0.9058    | 0.5291    | 0.09278   | 397.9     | 7.031     | 50.06     | 30.7      | 37.93     | 4.274     | 0.5536    | 0.9305    |
| 66        | 0.9062    | 0.5239    | 0.0929    | 397.9     | 7.03      | 50.06     | 30.7      | 37.93     | 4.277     | 0.5542    | 0.9299    |
| 67        | 0.9066    | 0.5567    | 0.1       | 398.0     | 7.008     | 50.01     | 30.72     | 37.91     | 4.339     | 0.55      | 0.8997    |
| 68        | 0.9071    | 0.5592    | 0.09836   | 398.0     | 7.01      | 50.02     | 30.72     | 37.91     | 4.342     | 0.5524    | 0.9013    |
| 69        | 0.9068    | 0.5123    | 0.1       | 397.9     | 6.963     | 49.97     | 30.67     | 37.86     | 4.295     | 0.5055    | 0.8698    |
| 70        | 0.9061    | 0.5152    | 0.1       | 397.9     | 6.966     | 49.97     | 30.68     | 37.87     | 4.298     | 0.5084    | 0.8717    |
| 71        | 0.9072    | 0.5197    | 0.1       | 397.9     | 6.971     | 49.98     | 30.68     | 37.87     | 4.302     | 0.5129    | 0.8748    |
| 72        | 0.9069    | 0.5142    | 0.0994    | 397.9     | 6.965     | 49.97     | 30.68     | 37.86     | 4.296     | 0.5068    | 0.86      |
| 73        | 0.9066    | 0.5202    | 0.09926   | 397.9     | 6.971     | 49.98     | 30.68     | 37.87     | 4.302     | 0.5127    | 0.8632    |
| 74        | 0.9054    | 0.5188    | 0.08852   | 397.9     | 6.97      | 49.98     | 30.68     | 37.87     | 4.301     | 0.5122    | 0.8706    |
| 75        | 0.9066    | 0.5265    | 0.1       | 397.9     | 6.973     | 49.98     | 30.68     | 37.87     | 4.303     | 0.5135    | 0.8697    |
| 76        | 0.9087    | 0.5787    | 0.1       | 397.9     | 6.993     | 49.99     | 30.68     | 37.86     | 4.313     | 0.5277    | 0.7845    |
'''
