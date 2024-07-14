# https://www.kaggle.com/competitions/otto-group-product-classification-challenge/overview

# [실습] acc : 0.89 이상

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd
import time

#1. 데이터
path = "C:\\ai5\\_data\\kaggle\\otto-group-product-classification-challenge\\"

train_csv = pd.read_csv(path + "train.csv", index_col=0)
test_csv = pd.read_csv(path + "test.csv", index_col=0)
submission_csv = pd.read_csv(path + "sampleSubmission.csv", index_col=0)

print(train_csv.isna().sum())       # 결측치 없음
print(test_csv.isna().sum())        # 결측치 없음

print(train_csv)                    # target이 Class_1 ... Class_9

encoder = LabelEncoder()
train_csv['target'] = encoder.fit_transform(train_csv['target'])    # 라벨링

print(train_csv)                    # target이 0 ... 8

x = train_csv.drop(['target'], axis=1)
y = train_csv['target']

print(x.shape)                      # (61878, 93), =input_dim
print(y.shape)                      # (61878,)
print(y)
# 1        0
# 2        0
# 3        0
# 4        0
# 5        0
#         ..
# 61874    8
# 61875    8
# 61876    8
# 61877    8
# 61878    8

print(pd.value_counts(y, sort=False))   # 라벨 별 카운트
# 0     1929
# 1    16122
# 2     8004
# 3     2691
# 4     2739
# 5    14135
# 6     2839
# 7     8464
# 8     4955

y_ohe = pd.get_dummies(y)           # OneHot

print(y_ohe.shape)                  # (61878, 9) ,9의 추가를 통해 OneHot 학인, =output_dim
print(y_ohe)
# 1      1  0  0  0  0  0  0  0  0
# 2      1  0  0  0  0  0  0  0  0
# 3      1  0  0  0  0  0  0  0  0
# 4      1  0  0  0  0  0  0  0  0
# 5      1  0  0  0  0  0  0  0  0
# ...   .. .. .. .. .. .. .. .. ..
# 61874  0  0  0  0  0  0  0  0  1
# 61875  0  0  0  0  0  0  0  0  1
# 61876  0  0  0  0  0  0  0  0  1
# 61877  0  0  0  0  0  0  0  0  1
# 61878  0  0  0  0  0  0  0  0  1

print(pd.value_counts(y, sort=False))   # 동일하게 나온다, 컴퓨터가 인식할때는 숫자로 인식해서?

x_train, x_test, y_train, y_test = train_test_split(x, y_ohe, train_size=0.9, random_state=3, stratify=y)

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
# scaler = MinMaxScaler()
scaler = StandardScaler()
# scaler = MaxAbsScaler()
# scaler = RobustScaler()

scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
test_csv = scaler.transform(test_csv)

print(x_train)
print(np.min(x_train), np.max(x_train))     # 0.0 1.0
print(np.min(x_test), np.max(x_test))       # 0.0 1.75

#2. 모델 구성
model = Sequential()
model.add(Dense(128, activation='relu', input_dim=93))
model.add(Dense(256, activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dense(1024, activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(9, activation='softmax'))

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
start = time.time()
es = EarlyStopping(monitor='val_loss', mode='min', patience=50, restore_best_weights=True)
model.fit(x_train, y_train, epochs=500, batch_size=60, verbose=1, validation_split=0.2, callbacks=[es])
end = time.time()

#4. 평가, 예측
loss = model.evaluate(x_test, y_test, verbose=1)
print('loss :', loss)
print('acc :', round(loss[1], 3))
print('time :', round(end-start, 2), '초')

# csv 파일 만들기 #
y_submit = model.predict(test_csv)
y_submit = np.round(y_submit)

submission_csv[['Class_1', 'Class_2', 'Class_3', 'Class_4', 'Class_5', 'Class_6', 'Class_7', 'Class_8', 'Class_9']] = y_submit  # 변경
submission_csv.to_csv(path + "sampleSubmission_0724_20.csv")

print(x)
'''
246 512 1024 1024 512 256 9 / 3 / epochs=300, batch_size=600 / loss : [0.5485449433326721, 0.7887847423553467] > 5.42856
256 126 126 126 126 1269 / 3 / epochs=500, batch_size=300 / loss : [0.5262376666069031, 0.8007434010505676] > 5.17476
256 126 64 32 16 9 / 3 / epochs=500, batch_size=60 / loss : [0.5155905485153198, 0.801874577999115] > 5.36005
256 126 64 64 64 9 / 3 / epochs=500, batch_size=100 / loss : [0.5188010931015015, 0.8009049892425537] > 4.91129
256 128 64 64 64 64 9 / 3 / epochs=500, batch_size=200 / loss : [0.5211756229400635, 0.7962185144424438] > 5.16181
256 126 64 64 9 / 3 / epochs=500, batch_size=100 / loss : [0.5281583666801453, 0.8004201650619507] > 5.58180
256 128 64 64 64 9 / 23 / epochs=500, batch_size=100 / loss : [0.5195449590682983, 0.7988041639328003] > 5.14556
256 128 64 64 64 9 / 3 / epochs=500, batch_size=100 / loss : [0.5182977318763733, 0.7991273403167725] > 5.33391
256 126 64 64 64 9 / 3 / epochs=500, batch_size=100 / loss : [0.5296603441238403, 0.8075307011604309] > 5.49480
256 512 256 128 64 32 16 9 / 3 / epochs=500, batch_size=128 / loss : [0.5356307029724121, 0.7918552160263062] > 5.55246
256 128 64 64 64 9 / 7777 / epochs=500, batch_size=100 / loss : [0.5333549976348877, 0.8005817532539368] > 5.38862
256 128 128 64 64 64 9 / 433 / epochs=500, batch_size=50 / loss : [0.501552402973175, 0.8070458769798279] > 5.35226
128 256 512 1024 512 256 128 64 9  / 3 / epochs=500, batch_size=60 / loss : [0.5428007245063782, 0.802359402179718] > 5.42862
++++++++++++++++++++++++++++++
MinMaxScaler / 128 256 512 1024 512 256 128 64 9  / 3 / epochs=500, batch_size=60 / loss : [0.5463241934776306, 0.7971881031990051] > 5.40218
StandardScaler / 128 256 512 1024 512 256 128 64 9  / 3 / epochs=500, batch_size=60 / loss : [0.5416991114616394, 0.8057530522346497] > 
'''
