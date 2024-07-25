# keras23_kaggle2_otto copt

# 과제  acc=0.89이상 만들어서 제출 캡쳐 이메일로 전송
# https://www.kaggle.com/competitions/otto-group-product-classification-challenge/data

import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
import time
from sklearn.metrics import r2_score, accuracy_score
from tensorflow.keras.callbacks import EarlyStopping

#1. 데이터
path = 'C:/AI5/_data/kaglle/Otto Group/'

train_csv = pd.read_csv(path + 'train.csv', index_col=0)
print(train_csv)    # [61878 rows x 94 columns]

test_csv = pd.read_csv(path + 'test.csv', index_col=0)
print(test_csv)     # [144368 rows x 93 columns]

submission_csv = pd.read_csv(path + 'sampleSubmission.csv', index_col=0)
print(submission_csv)   # [144368 rows x 9 columns]

print(train_csv.columns)

print(train_csv.isna().sum())   # 0
print(test_csv.isna().sum())    # 0

# label encoder
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
train_csv['target'] = le.fit_transform(train_csv['target'])
# print(train_csv['target'])

x = train_csv.drop(['target'], axis=1)
print(x)
y = train_csv['target']
print(y.shape)

y_ohe = pd.get_dummies(y)

x_train, x_test, y_train, y_test = train_test_split(x,y_ohe,train_size=0.8,
                                                    random_state=3752,
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
model.add(Dense(128, activation='relu', input_dim=93))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(9, activation='softmax'))

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
start = time.time()

es = EarlyStopping(
    monitor = 'val_loss',
    mode = 'min',
    patience = 77,
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


y_submit = model.predict(test_csv)      # round 꼭 넣기
print(y_submit)
print(y_submit.shape)     

#################  submission.csv 만들기 // count 컬럼에 값만 넣어주면 된다 ######
#submission_csv['target'] = y_submit
print(submission_csv)
print(submission_csv.shape)

submission_csv[['Class_1', 'Class_2', 'Class_3', 'Class_4', 'Class_5', 'Class_6', 'Class_7', 'Class_8', 'Class_9']] = y_submit

# for i in range(9) :
#     submission_csv['Class_' + str(i+1)] = y_submit[:,i].astype('int')


submission_csv.to_csv(path + "submission_0725_1620.csv")

print('로스 :', loss)
print("acc :", round(loss[1],3))

# MinMaxScaler
# 로스 : [0.541971743106842, 0.7923400402069092]
# acc : 0.792

# StandardScaler
# 로스 : [0.5397934913635254, 0.7953296899795532]
# acc : 0.795

# MaxAbsScaler
# 로스 : [0.5393970012664795, 0.7954104542732239]
# acc : 0.795

# RobusterScaler
# 로스 : [0.5412541031837463, 0.7942792773246765]
# acc : 0.794
