import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder

#1. 데이터
path = "C:\\ai5\\_data\\kaggle\\otto-group-product-classification-challenge\\"

train_csv = pd.read_csv(path + "train.csv", index_col=0)
test_csv = pd.read_csv(path + "test.csv", index_col=0)
submission_csv = pd.read_csv(path + "sampleSubmission.csv", index_col=0)

encoder = LabelEncoder()
train_csv['target'] = encoder.fit_transform(train_csv['target'])    # 라벨링

x = train_csv.drop(['target'], axis=1)
y = train_csv['target']

# x.boxplot()        # 이상치 확인, Population 에 이상치가 있다
# x.plot.box()     # 동일
# plt.show()

# print(x.info())    # 모두 non-null
# print(x.describe)

# df['Population'].boxplot()
# plt.show()        # 시리즈에서 안된다, AttributeError: 'Series' object has no attribute 'boxplot'

# x['feat_73'].plot.box()
# plt.show()        # 된다

# x['feat_73'].hist(bins=50)
# plt.show()        # 된다

# y.hist(bins=50)
# plt.show()        # 된다

########## Population 로그 변환 ##########
x['feat_73'] = np.log1p(x['feat_73']) # 지수변환 np.expm1

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
score :  0.47953740861221794
r2 score :  0.47953740861221794

y만 로그 변환
score :  0.4409163563584556
r2 score :  0.4409163563584556

feat_73만 로그 변환
score :  0.47953740861221794
r2 score :  0.47953740861221794

로그변환O
score :  0.4409163563584556
r2 score :  0.4409163563584556

##### LinearRegression #####
로그변환X
score :  0.5336291063667085
r2 score :  0.5336291063667085

y만 로그 변환
score :  0.4640542258601543
r2 score :  0.4640542258601543

feat_73만 로그 변환
score :  0.5345309871450357
r2 score :  0.5345309871450357

로그변환O
score :  0.46464147866138983
r2 score :  0.46464147866138983

'''
