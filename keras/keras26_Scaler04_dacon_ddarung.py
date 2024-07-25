# keras19_EarlyStopping4_dacon_ddarung copy

# https://dacon.io/competitions/open/235576/overview/description

import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import time

#1. 데이터
path = "C:/AI5/_data/dacon/따릉이/"        # 경로지정  상대경로

train_csv = pd.read_csv(path + "train.csv", index_col=0)   # . = 루트 = AI5 폴더,  index_col=0 첫번째 열은 데이터가 아니라고 해줘
print(train_csv)     # [1459 rows x 10 columns]

test_csv = pd.read_csv(path + "test.csv", index_col=0)
print(test_csv)     # [715 rows x 9 columns]

submission_csv = pd.read_csv(path + "submission.csv", index_col=0)
print(submission_csv)       #[715 rows x 1 columns],    NaN = 빠진 데이터
# 항상 오타, 경로 , shape 조심 확인 주의

print(train_csv.shape)  #(1459, 10)
print(test_csv.shape)   #(715, 9)
print(submission_csv.shape)     #(715, 1)

print(train_csv.columns)
# # ndex(['hour', 'hour_bef_temperature', 'hour_bef_precipitation',
#        'hour_bef_windspeed', 'hour_bef_humidity', 'hour_bef_visibility',
#        'hour_bef_ozone', 'hour_bef_pm10', 'hour_bef_pm2.5', 'count'],
#       dtype='object')
# x 는 결측치가 있어도 괜찮지만 y 는 있으면 안된다

train_csv.info()

################### 결측치 처리 1. 삭제 ###################
# print(train_csv.isnull().sum())
print(train_csv.isna().sum())   # 결측치 확인

train_csv = train_csv.dropna()   # 결측치 삭제
print(train_csv.isna().sum())    # 삭제 뒤 결측치 확인
print(train_csv)        #[1328 rows x 10 columns]
print(train_csv.isna().sum())
print(train_csv.info())

print(test_csv.info())
#  test_csv 는 결측치 삭제 불가, test_csv 715 와 submission 715 가 같아야 한다.
#  그래서 결측치 삭제하지 않고, 데이터의 평균 값을 넣어준다.

test_csv = test_csv.fillna(test_csv.mean())     #컬럼끼리만 평균을 낸다
print(test_csv.info())

x = train_csv.drop(['count'], axis=1)           # drop = 컬럼 하나를 삭제할 수 있다.   # axis=1 이면 열, 0 이면 행  카운트 열을 지워라
print(x)        #[1328 rows x 9 columns]
y = train_csv['count']         # 'count' 컬럼만 넣어주세요
print(y.shape)   # (1328,)

x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.9822,
                                                    random_state= 5757)


from sklearn.preprocessing import MinMaxScaler, StandardScaler

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
# scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
scaler = RobustScaler()

scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
test_csv = scaler.transform(test_csv)

print('x_train :', x_train)
print(np.min(x_train), np.max(x_train))
print(np.min(x_test), np.max(x_test))


#2. 모델구성
model = Sequential()
model.add(Dense(10, input_dim=9))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
start = time.time()

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(
    monitor = 'val_loss',
    mode = 'min',
    patience =20,
    restore_best_weights=True
)
hist = model.fit(x_train, y_train, validation_split=0.2,
           epochs=1000, batch_size=32,
           callbacks=[es])
end = time.time()

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('로스 :', loss)

y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print("r2스코어 :", r2)
print("걸린시간 :", round(end-start,2),"초")

# print("================ hist ===============")
# print(hist)
# print("================ hist.history ========")
# print(hist.history)
# print("================ loss ================")
# print(hist.history['loss'])
# print("=============== val_loss =============")
# print(hist.history['val_loss'])
# print("======================================")

# import matplotlib.pyplot as plt
# plt.rcParams['font.family'] ='Malgun Gothic'
# plt.figure(figsize=(9,6))       # 그림판 사이즈
# plt.plot(hist.history['loss'], c='red', label='loss')
# plt.plot(hist.history['val_loss'], c='blue', label='val_loss')
# plt.legend(loc='upper right')
# plt.title('따릉이 loss')
# plt.xlabel('epochs')
# plt.ylabel('loss')
# plt.grid()          #모눈종이
# plt.show()

# MinMaxScaler
# 로스 : 1176.9427490234375
# r2스코어 : 0.8239208138560189
# 걸린시간 : 2.25 초

# StandardScaler
# 로스 : 1124.8985595703125
# r2스코어 : 0.8317069934221893
# 걸린시간 : 1.02 초

# MaxAbsScaler
# 로스 : 1123.9234619140625
# r2스코어 : 0.8318528481008873
# 걸린시간 : 2.28 초

#RobustScaler
# 로스 : 1046.8455810546875
# r2스코어 : 0.8433842708143588
# 걸린시간 : 1.1 초