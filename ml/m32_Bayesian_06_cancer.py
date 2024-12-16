import numpy as np
import time
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score
from sklearn.preprocessing import MinMaxScaler
import xgboost as xgb
from xgboost import XGBClassifier
from bayes_opt import BayesianOptimization
import warnings
warnings.filterwarnings('ignore')

# 1. 데이터
x, y = load_breast_cancer(return_X_y=True)

random_state=777
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, stratify=y, random_state=random_state)

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
{'target': 0.9824561403508771, 'params': {'learning_rate': 0.08078474307324947, 'max_depth': 7.525186390258969}}
100 번 걸린시간 : 17.65 초

{'target': 0.9912280701754386, 'params': {'colsample_bytree': 0.5, 'learning_rate': 0.1, 'max_bin': 162.06656904731443, 'max_depth': 3.0, 'min_child_samples': 87.42970754123296, 'min_child_weight': 1.0, 'num_leaves': 24.0, 'reg_alpha': 0.01, 'reg_lambda': -0.001, 'subsample': 1.0}}
100 번 걸린시간 : 25.84 초
'''

'''
|   iter    |  target   | colsam... | learni... |  max_bin  | max_depth | min_ch... | min_ch... | num_le... | reg_alpha | reg_la... | subsample |
-------------------------------------------------------------------------------------------------------------------------------------------------
| 1         | 0.9737    | 0.7716    | 0.07317   | 17.29     | 5.312     | 80.06     | 3.367     | 25.67     | 4.881     | 2.453     | 0.9367    |
| 2         | 0.9123    | 0.5258    | 0.006988  | 126.2     | 4.426     | 43.16     | 21.35     | 39.07     | 23.35     | 9.6       | 0.9999    |
| 3         | 0.9211    | 0.8632    | 0.04894   | 71.29     | 3.738     | 18.03     | 12.96     | 25.14     | 49.78     | 2.012     | 0.9412    |
| 4         | 0.9474    | 0.6226    | 0.01243   | 299.8     | 7.958     | 34.75     | 6.807     | 24.06     | 28.48     | 4.818     | 0.6706    |
| 5         | 0.9561    | 0.6324    | 0.03542   | 461.1     | 9.882     | 195.5     | 3.095     | 36.93     | 43.34     | 5.179     | 0.8577    |
| 6         | 0.9561    | 0.7029    | 0.01963   | 301.7     | 6.193     | 29.56     | 9.169     | 28.42     | 28.1      | 7.836     | 0.7818    |
| 7  *      | 0.9825    | 0.7654    | 0.06237   | 16.76     | 7.773     | 89.1      | 5.31      | 29.17     | 2.774     | 3.112     | 0.5506    |
| 8         | 0.9474    | 0.7458    | 0.06304   | 297.7     | 8.203     | 49.51     | 27.19     | 31.08     | 37.3      | 2.294     | 0.5519    |
| 9         | 0.9561    | 0.9757    | 0.00779   | 9.0       | 9.994     | 128.1     | 1.0       | 28.14     | 0.01      | 0.3234    | 0.5       |
| 10        | 0.9825    | 0.7706    | 0.09235   | 216.4     | 9.567     | 102.0     | 7.163     | 27.84     | 5.36      | 9.867     | 0.8579    |
| 11        | 0.6316    | 0.5       | 0.1       | 14.34     | 10.0      | 91.2      | 36.47     | 40.0      | 0.01      | 10.0      | 0.5       |
| 12        | 0.9561    | 0.5877    | 0.02233   | 300.2     | 5.339     | 28.82     | 9.488     | 34.42     | 26.47     | 8.216     | 0.6965    |
| 13        | 0.9561    | 0.8633    | 0.08679   | 301.9     | 9.56      | 48.58     | 29.79     | 32.05     | 38.69     | 1.129     | 0.683     |
| 14        | 0.9737    | 0.6892    | 0.02965   | 210.7     | 5.074     | 136.1     | 8.456     | 30.53     | 19.21     | 5.217     | 0.9306    |
| 15        | 0.9649    | 0.6741    | 0.08777   | 460.7     | 7.846     | 192.9     | 3.804     | 33.75     | 44.23     | 6.163     | 0.8937    |
| 16        | 0.9737    | 0.9958    | 0.08916   | 251.1     | 10.0      | 124.1     | 9.471     | 26.81     | 6.839     | 8.983     | 0.9945    |
| 17        | 0.6316    | 0.8808    | 0.05217   | 226.1     | 8.203     | 114.0     | 44.13     | 30.16     | 13.48     | 2.811     | 0.7877    |
| 18        | 0.9737    | 0.5       | 0.1       | 184.1     | 10.0      | 118.2     | 1.0       | 24.0      | 0.01      | 10.0      | 1.0       |
| 19        | 0.9737    | 0.5013    | 0.1       | 234.4     | 10.0      | 158.6     | 1.0       | 24.0      | 0.01      | 10.0      | 1.0       |
| 20        | 0.9649    | 0.5128    | 0.09992   | 188.6     | 3.0       | 164.5     | 1.0       | 24.39     | 2.696     | 10.0      | 1.0       |
| 21        | 0.9561    | 0.9704    | 0.1       | 281.2     | 8.406     | 148.3     | 1.0       | 24.0      | 0.01      | 10.0      | 1.0       |
| 22        | 0.9737    | 0.5       | 0.1       | 193.4     | 9.399     | 72.67     | 1.0       | 24.0      | 0.01      | 10.0      | 1.0       |
| 23        | 0.9561    | 0.6139    | 0.1       | 217.0     | 7.141     | 170.1     | 1.0       | 24.0      | 41.65     | 10.0      | 0.8505    |
| 24        | 0.9649    | 0.5       | 0.1       | 189.3     | 3.0       | 98.54     | 1.0       | 24.0      | 35.67     | 10.0      | 1.0       |
| 25        | 0.6316    | 1.0       | 0.001     | 174.5     | 6.002     | 142.8     | 1.0       | 40.0      | 37.94     | 10.0      | 0.5011    |
| 26        | 0.9561    | 0.5       | 0.1       | 250.5     | 3.0       | 146.6     | 1.0       | 24.0      | 34.77     | 10.0      | 1.0       |
| 27        | 0.9737    | 0.5       | 0.1       | 214.4     | 3.0       | 195.5     | 1.0       | 24.0      | 1.856     | -0.001    | 1.0       |
| 28        | 0.9561    | 0.5       | 0.1       | 217.1     | 3.0       | 68.28     | 1.0       | 24.0      | 32.82     | 10.0      | 1.0       |
| 29        | 0.6316    | 0.9183    | 0.001     | 249.4     | 6.773     | 192.9     | 1.0       | 40.0      | 21.97     | 6.837     | 0.8513    |
| 30        | 0.6316    | 1.0       | 0.001     | 37.87     | 3.0       | 105.4     | 1.0       | 24.0      | 22.15     | -0.001    | 0.5       |
| 31        | 0.9737    | 0.5       | 0.1       | 209.2     | 10.0      | 168.5     | 20.28     | 24.0      | 0.01      | -0.001    | 1.0       |
| 32        | 0.9649    | 0.5       | 0.1       | 181.3     | 3.0       | 67.9      | 1.0       | 40.0      | 27.47     | -0.001    | 1.0       |
| 33        | 0.9737    | 0.5       | 0.1       | 184.0     | 5.703     | 199.8     | 14.96     | 24.0      | 0.01      | -0.001    | 1.0       |
| 34        | 0.9649    | 0.5       | 0.1       | 281.3     | 4.082     | 114.8     | 1.0       | 24.0      | 23.99     | 10.0      | 1.0       |
| 35 *      | 0.9912    | 0.5       | 0.1       | 162.1     | 3.0       | 87.43     | 1.0       | 24.0      | 0.01      | -0.001    | 1.0       |
'''
