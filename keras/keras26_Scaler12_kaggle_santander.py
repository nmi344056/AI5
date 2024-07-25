# keras23_kaggle1_santander_customer copy

# https://www.kaggle.com/competitions/santander-customer-transaction-prediction/data

# 맹그러
# 다중분류인줄 알았더니 이진분류였다!!!
# 다중분류 다시 찾겠노라!!!

import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
import time
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score, accuracy_score

#1. 데이터
path = 'C:/AI5/_data/kaglle/santander_customer/'

train_csv = pd.read_csv(path + 'train.csv', index_col=0)
print(train_csv)    # [200000 rows x 201 columns]

test_csv = pd.read_csv(path + 'test.csv', index_col=0)
print(test_csv)     # [200000 rows x 200 columns]

submission_csv = pd.read_csv(path + "sample_submission.csv", index_col=0)
print(submission_csv)      # [200000 rows x 1 columns]

print(train_csv.shape)      # (200000, 201)
print(test_csv.shape)       # (200000, 200)
print(submission_csv.shape) # (200000, 1)

print(train_csv.columns)

# train_csv.info()    
# test_csv.info()     


x = train_csv.drop(['target'], axis=1)
print(x)
y = train_csv['target']         # 'count' 컬럼만 넣어주세요
print(y.shape)   

x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.8,
                                                    random_state=632,
                                                    stratify=y)


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
model.add(Dense(128, activation='relu', input_dim=200))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(2, activation='relu')) #중간에 sigmoid 넣어줄수있다.

model.add(Dense(1, activation='sigmoid'))

#3. 컴파일, 훈련
# model.compile(loss='mse', optimizer='adam', metrics=['accuracy', 'acc', 'mse'])  # 매트릭스에 애큐러시를 넣으면 반올림해준다.
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
start = time.time()

es = EarlyStopping(
    monitor = 'val_loss',
    mode = 'min',
    patience = 100,
    restore_best_weights=True
)
hist = model.fit(x_train, y_train, epochs=2000, batch_size=128,
                 validation_split=0.2,
                 callbacks=[es]
                 )
end = time.time()

#4. 평가, 예측
loss = model.evaluate(x_test, y_test,
                      verbose=1)
print('로스 : ', loss[0])
print("acc : ", round(loss[1],3))  # 애큐러시, 3자리 반올림 

y_pred = model.predict(x_test)
print(y_pred)
y_pred = np.round(y_pred)
print(y_pred)

accuracy_score = accuracy_score(y_test, y_pred)  
# r2 = r2_score(y_test, y_predict)
print("acc_score : ", accuracy_score)
print("걸린시간 : ", round(end - start , 2),"초")

y_submit = np.round(model.predict(test_csv))      # round 꼭 넣기
print(y_submit)
print(y_submit.shape)     

#################  submission.csv 만들기 // count 컬럼에 값만 넣어주면 된다 ######
submission_csv['target'] = y_submit
print(submission_csv)
print(submission_csv.shape)

submission_csv.to_csv(path + "submission_0725_1552.csv")

print('로스 :', loss)
print("acc :", round(loss[1],3))

# 로스 : [0.23520657420158386, 0.9101123809814453]
# acc : 0.91

# MinMaxScaler 적용
# 로스 : [0.2344352751970291, 0.9135249853134155]
# acc : 0.914

# StandardScaler
# 로스 : [0.24622783064842224, 0.8995000123977661]
# acc : 0.9

# MaxAbsScaler
# 로스 : [0.2419477105140686, 0.9108999967575073]
# acc : 0.911

# RobustScaler
# 로스 : [0.24546508491039276, 0.8995000123977661]
# acc : 0.9