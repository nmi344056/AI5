import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

#1. 데이터
datasets = load_diabetes()
df = pd.DataFrame(datasets.data, columns=datasets.feature_names)
print(df)

df['target'] = datasets.target

# df.boxplot()        # 이상치 확인, Population 에 이상치가 있다
# df.plot.box()       # 동일
# plt.show()

print(df.info())      # 모두 non-null
print(df.describe)

# df['target'].hist(bins=50)
# plt.show()

x = df.drop(['target'], axis=1).copy()
y = df['target']

########## Population 로그 변환 ##########

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
score :  0.4703952770087839
r2 score :  0.4703952770087839

y 로그 변환
score :  0.40413559698216805
r2 score :  0.40413559698216805

##### LinearRegression #####
로그변환X
score :  0.4626336507981068
r2 score :  0.4626336507981068

y 로그 변환
score :  0.40454182335284894
r2 score :  0.40454182335284894

'''
