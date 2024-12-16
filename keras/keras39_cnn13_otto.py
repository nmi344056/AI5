# https://www.kaggle.com/competitions/otto-group-product-classification-challenge/overview

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
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

x = x.to_numpy()
x = x/255.
x = x.reshape(61878, 93, 1, 1)
print(x.shape)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.9, random_state=3, stratify=y)

scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
# scaler = RobustScaler()

#2. 모델 구성
model = Sequential()
model.add(Conv2D(64, (3,3), activation='relu', input_shape=(93, 1, 1), strides=1, padding='same'))
model.add(Dropout(0.3))
model.add(Conv2D(filters=64, kernel_size=(3,3), activation='relu', strides=1, padding='same'))
model.add(Dropout(0.2))
model.add(Conv2D(32, (2,2), activation='relu', strides=1, padding='same'))
model.add(Flatten())

model.add(Dense(units=32, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(units=16, input_shape=(32,), activation='relu'))
model.add(Dense(1, activation='linear'))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['acc'])

es = EarlyStopping(monitor='val_loss', mode='min', patience=50, verbose=1, restore_best_weights=True,)

########## mcp 세이브 파일명 만들기 시작 ##########
import datetime
date = datetime.datetime.now()
print(date)
print(type(date))

date = date.strftime("%m%d.%H%M")
print(date)
print(type(date))

path = './_save/keras39/'
filename = '{epoch:04d}_valloss_{val_loss:.4f}.hdf5'
filepath = "".join([path, 'k39_13_otto_diabetes_date_', date, '_epo_', filename])

########## mcp 세이브 파일명 만들기 끝 ##########

mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1, save_best_only=True, filepath=filepath)

start = time.time()
hist = model.fit(x_train, y_train, epochs=1, batch_size=60, validation_split=0.2, callbacks=[es, mcp])
end = time.time()

#4. 평가, 예측
loss = model.evaluate(x_test, y_test, verbose=1)
print('loss :', loss)
print('acc :', round(loss[1], 3))
print('time :', round(end-start, 2), '초')

# # csv 파일 만들기 #
# y_submit = model.predict(test_csv)
# y_submit = np.round(y_submit)

# submission_csv[['Class_1', 'Class_2', 'Class_3', 'Class_4', 'Class_5', 'Class_6', 'Class_7', 'Class_8', 'Class_9']] = y_submit  # 변경
# submission_csv.to_csv(path + "sampleSubmission_0724_20.csv")

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

loss : [0.5208882689476013, 0.8039754629135132]
acc : 0.804


'''
