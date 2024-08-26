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
path = "C:\\ai5\\_data\\kaggle\\bike-sharing-demand\\"
# path = "C://ai5//_data//bike-sharing-demand//"
# path = "C://ai5/_data/bike-sharing-demand/"

train_csv = pd.read_csv(path + "train.csv", index_col=0)
test_csv = pd.read_csv(path + "test.csv", index_col=0)
sampleSubmission = pd.read_csv(path + "sampleSubmission.csv", index_col=0)

print(train_csv.columns)
# Index(['season', 'holiday', 'workingday', 'weather', 'temp', 'atemp',
#        'humidity', 'windspeed', 'casual', 'registered', 'count'],    
#       dtype='object')

x = train_csv.drop(['casual', 'registered', 'count'], axis=1)
print(x)                            # [10886 rows x 8 columns]
y = train_csv['count']
print(y)
print(y.shape)                      # (10886,)

# x.boxplot()         # 이상치 확인, Population 에 이상치가 있다
# x.plot.box()      # 동일
# plt.show()

# print(x.info())  # 모두 non-null
# print(x.describe)

# df['Population'].boxplot()
# plt.show()        # 시리즈에서 안된다, AttributeError: 'Series' object has no attribute 'boxplot'

# x['windspeed'].plot.box()
# plt.show()        # 된다

# x['windspeed'].hist(bins=50)
# plt.show()

# y.hist(bins=50)
# plt.show()

########## Population 로그 변환 ##########
# x['windspeed'] = np.log1p(x['windspeed']) # 지수변환 np.expm1

# [실습] 만들기
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=1234)

########## y 로그 변환 ##########
# y_train = np.log1p(y_train)
# y_test = np.log1p(y_test)
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
score :  0.32454610690908825
r2 score :  0.32454610690908825

y만 로그 변환
score :  0.2962019047786678
r2 score :  0.2962019047786678

windspeed만 로그 변환
score :  0.32455302892822135
r2 score :  0.32455302892822135

로그변환O
score :  0.2962019047786678
r2 score :  0.2962019047786678

##### LinearRegression #####
로그변환X
score :  0.25848766228349007
r2 score :  0.25848766228349007

y만 로그 변환
score :  0.2513527220315678
r2 score :  0.2513527220315678

windspeed만 로그 변환
score :  0.2585271734411353
r2 score :  0.2585271734411353

로그변환O
score :  0.25140101689974914
r2 score :  0.25140101689974914

'''
