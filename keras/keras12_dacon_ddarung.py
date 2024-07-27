# https://dacon.io/competitions/open/235576/data

import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

#1. 데이터
path = "./_data/dacon/따릉이/"

train_csv = pd.read_csv(path + "train.csv", index_col=0)
print(train_csv)            # [1459 rows x 11 columns] / [1459 rows x 10 columns]

test_csv = pd.read_csv(path + "test.csv", index_col=0)
print(test_csv)             # [715 rows x 9 columns]

submission_csv = pd.read_csv(path + "submission.csv", index_col=0)
print(submission_csv)       # [715 rows x 1 columns]

print(train_csv.shape)      # (1459, 10)
print(test_csv.shape)       # (715, 9)
print(submission_csv.shape) # (715, 1)

print(train_csv.columns)
# Index(['hour', 'hour_bef_temperature', 'hour_bef_precipitation',
#        'hour_bef_windspeed', 'hour_bef_humidity', 'hour_bef_visibility',
#        'hour_bef_ozone', 'hour_bef_pm10', 'hour_bef_pm2.5', 'count'],
#       dtype='object')

print(train_csv.info())     # DataFrame의 기본 정보 출력
'''
 #   Column                  Non-Null Count  Dtype
---  ------                  --------------  -----
 0   hour                    1459 non-null   int64
 1   hour_bef_temperature    1457 non-null   float64
 2   hour_bef_precipitation  1457 non-null   float64
 3   hour_bef_windspeed      1450 non-null   float64
 4   hour_bef_humidity       1457 non-null   float64
 5   hour_bef_visibility     1457 non-null   float64
 6   hour_bef_ozone          1383 non-null   float64
 7   hour_bef_pm10           1369 non-null   float64
 8   hour_bef_pm2.5          1342 non-null   float64
 9   count                   1459 non-null   float64
 '''
 
########## 결측치 처리 1. 삭제 ##########
# print(train_csv.isnull().sum())   # 결측치 전체 개수 출력
'''
hour                        0
hour_bef_temperature        2
hour_bef_precipitation      2
hour_bef_windspeed          9
hour_bef_humidity           2
hour_bef_visibility         2
hour_bef_ozone             76
hour_bef_pm10              90
hour_bef_pm2.5            117
count                       0
'''
print(train_csv.isna().sum())   # 동일

train_csv = train_csv.dropna()  # 결측치 포함 행 삭제
print(train_csv.isna().sum())   # 결측치 삭제 확인
'''
hour                      0
hour_bef_temperature      0
hour_bef_precipitation    0
hour_bef_windspeed        0
hour_bef_humidity         0
hour_bef_visibility       0
hour_bef_ozone            0
hour_bef_pm10             0
hour_bef_pm2.5            0
count                     0
'''
print(train_csv)            # [1328 rows x 10 columns]

print(train_csv.info())
'''
#   Column                  Non-Null Count  Dtype
---  ------                  --------------  -----
 0   hour                    1328 non-null   int64
 1   hour_bef_temperature    1328 non-null   float64
 2   hour_bef_precipitation  1328 non-null   float64
 3   hour_bef_windspeed      1328 non-null   float64
 4   hour_bef_humidity       1328 non-null   float64
 5   hour_bef_visibility     1328 non-null   float64
 6   hour_bef_ozone          1328 non-null   float64
 7   hour_bef_pm10           1328 non-null   float64
 8   hour_bef_pm2.5          1328 non-null   float64
 9   count                   1328 non-null   float64
 '''
print(test_csv.info())
'''
 #   Column                  Non-Null Count  Dtype
---  ------                  --------------  -----
 0   hour                    715 non-null    int64
 1   hour_bef_temperature    714 non-null    float64
 2   hour_bef_precipitation  714 non-null    float64
 3   hour_bef_windspeed      714 non-null    float64
 4   hour_bef_humidity       714 non-null    float64
 5   hour_bef_visibility     714 non-null    float64
 6   hour_bef_ozone          680 non-null    float64
 7   hour_bef_pm10           678 non-null    float64
 8   hour_bef_pm2.5          679 non-null    float64
 '''
 
test_csv = test_csv.fillna(test_csv.mean())     # fillna=결측치 채우기, mean=컬럼별 평균값으로
print(test_csv.info())
'''
 #   Column                  Non-Null Count  Dtype
---  ------                  --------------  -----
 0   hour                    715 non-null    int64
 1   hour_bef_temperature    715 non-null    float64
 2   hour_bef_precipitation  715 non-null    float64
 3   hour_bef_windspeed      715 non-null    float64
 4   hour_bef_humidity       715 non-null    float64
 5   hour_bef_visibility     715 non-null    float64
 6   hour_bef_ozone          715 non-null    float64
 7   hour_bef_pm10           715 non-null    float64
 8   hour_bef_pm2.5          715 non-null    float64
'''
x = train_csv.drop(['count'], axis=1)
print(x)                    # [1328 rows x 9 columns]
y = train_csv['count']
print(y)
print(y.shape)              # (1328,)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=4343)

#2. 모델구성
model = Sequential()
model.add(Dense(64, input_dim=9))
model.add(Dense(64))
model.add(Dense(64))
model.add(Dense(64))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=500, batch_size=23)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print("loss : ", loss)

y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print("r2 score : ", r2)

'''
64 64 64 64 / train_size=0.8, random_state=434 / epochs=500, batch_size=23 / loss :  3083.748779296875 / r2 score :  0.5581963603746334
64 64 64 64 / train_size=0.8, random_state=4343 / epochs=500, batch_size=23 / loss :  2202.378173828125 / r2 score :  0.6335449902514239
64 32 32 16 / train_size=0.8, random_state=4343 / epochs=500, batch_size=23 / loss :  2220.0771484375 / r2 score :  0.6306000388327853
'''

y_submit = model.predict(test_csv)
print(y_submit)
print(y_submit.shape)           # (715, 1)

########## submission.csv 만들기 (count컬럼에 값만 넣으면 된다) ##########
submission_csv['count'] = y_submit
print(submission_csv)           # [715 rows x 1 columns]
print(submission_csv.shape)     # (715, 1)

submission_csv.to_csv(path + "submission_0716_1630.csv")

print("++++++++++++++++++++")
print("loss : ", loss, "/ r2 score : ", r2)

'''
#[과제]  고도화해서 DACON에 10번 업로드


'''
