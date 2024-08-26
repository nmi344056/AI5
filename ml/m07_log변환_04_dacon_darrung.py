import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

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

# x.boxplot()               # 이상치 확인, Population 에 이상치가 있다
# df.plot.box()             # 동일
# plt.show()

# print(x.info())           # 모두 non-null
# print(x.describe)

# df['Population'].boxplot()
# plt.show()                # 시리즈에서 안된다, AttributeError: 'Series' object has no attribute 'boxplot'

# train_csv['hour_bef_pm10'].plot.box()
# plt.show()                # 된다

# train_csv['hour_bef_pm10'].hist(bins=50)
# plt.show()

# y.hist(bins=50)
# plt.show()

########## Population 로그 변환 ##########
x['hour_bef_pm10'] = np.log1p(x['hour_bef_pm10']) # 지수변환 np.expm1

# [실습] 만들기
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=1234)

########## y 로그 변환 ##########
y_train = np.log1p(y_train)
y_test = np.log1p(y_test)
################################

#2. 모델 구성
model = RandomForestRegressor(random_state=1234, max_depth=5, min_samples_split=3)
# model = LinearRegression()

#3. 훈련
model.fit(x_train, y_train)

#4. 평가,예측
score = model.score(x_test, y_test)     # score = r2_score
print("score : ", score)

y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print("r2 score : ", r2)

'''
RandomForestRegressor
로그변환X
score :  0.7766179166232445
r2 score :  0.7766179166232445

y만 로그 변환
score :  0.7849157517950008
r2 score :  0.7849157517950008

hour_bef_pm10만 로그 변환
score :  0.7766991762182309
r2 score :  0.7766991762182309

로그변환O
score :  0.784927906562371
r2 score :  0.784927906562371

##### LinearRegression #####
로그변환X
score :  0.6006145556398459
r2 score :  0.6006145556398459

y만 로그 변환
score :  0.5855336673037024
r2 score :  0.5855336673037024

hour_bef_pm10만 로그 변환
score :  0.598417235392551
r2 score :  0.598417235392551

로그변환O
score :  0.5795221097373962
r2 score :  0.5795221097373962

'''
