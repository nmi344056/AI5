import numpy as np
import pandas as pd
import time
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Concatenate, concatenate
from tensorflow.keras.layers import SimpleRNN, LSTM, GRU, Conv1D, Conv2D, Flatten, Dropout, MaxPooling1D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split

#1. 데이터
path = "C:\\ai5\\_data\\중간고사데이터\\"
NAVER_csv = pd.read_csv(path + "NAVER 240816.csv", index_col=0 , encoding='cp949', thousands=',')
HYBE_csv = pd.read_csv(path + "하이브 240816.csv", index_col=0 , encoding='cp949', thousands=',')
seongu_csv = pd.read_csv(path + "성우하이텍 240816.csv", index_col=0 , encoding='cp949', thousands=',')

print(NAVER_csv.shape, HYBE_csv.shape, seongu_csv.shape)    # (5390, 16) (948, 16) (7058, 16)

print(NAVER_csv.columns)
# Index(['시가', '고가', '저가', '종가', '전일비', 'Unnamed: 6', '등락률', '거래량', '금액(백만)',
#        '신용비', '개인', '기관', '외인(수량)', '외국계', '프로그램', '외인비'],

# print(NAVER_csv)
'''
일자         시가      고가       저가      종가      전일비   Unnamed: 6   등락률    거래량    금액(백만)   신용비   개인    기관   외인(수량)   외국계     프로그램   외인비
2024/08/16   159,200   159,200   157,000   159,000   ▲       1,700        1.08     348,137   55,134      0.00     0      0      0           -106,245   -33,372   42.88
'''

# NAVER_csv = NAVER_csv.loc[::-1]       # 역순 정렬 
# HYBE_csv = HYBE_csv.loc[::-1]         # 역순 정렬  
# seongu_csv = seongu_csv.loc[::-1]     # 역순 정렬 

NAVER_csv = NAVER_csv[:948]
seongu_csv = seongu_csv[:948]
seongu_csv = seongu_csv[:948]

x_n = NAVER_csv.drop(['전일비', 'Unnamed: 6', '금액(백만)', '신용비', '개인', '기관', '외인(수량)', '외국계', '프로그램', '외인비'], axis=1)
x_h = HYBE_csv.drop(['전일비', 'Unnamed: 6', '금액(백만)', '신용비', '개인', '기관', '외인(수량)', '외국계', '프로그램', '외인비'], axis=1)
y_u = seongu_csv['종가']
print(x_n.shape, x_h.shape, y_u.shape)          # (948, 6) (948, 6) (948,)

# print(x_n)
# print(x_h)
# print(y_u)

x_n.iloc[0] = [159200.00, 159200.00, 157000.00, 157500.00, 0.13, 814631.00]
x_h.iloc[0] = [164700.00, 168600.00, 163500.00, 166400.00, 2.02, 188910.00]

# print(x_n.iloc[0])
# print(x_h.iloc[0])

print(x_n.info())
print(x_n.isnull().sum())
print(x_h.info())
print(x_h.isnull().sum())

size = 20

def split_x(dataset, size):
    aaa = []
    for i in range(len(dataset) - size + 1):
        subset = dataset[i : (i + size)]
        aaa.append(subset)
    return np.array(aaa)

x_n = split_x(x_n, size)
x_h = split_x(x_h, size)
print(x_n.shape, x_h.shape)                     # (929, 20, 6) (929, 20, 6)

xn_predict = x_n[:1]
xh_predict = x_h[:1]
print(xn_predict.shape, xh_predict.shape)       # (1, 20, 6) (1, 20, 6)

# print(xn_predict)

x_n = x_n[1:]
x_h = x_h[1:]
y_u = y_u[:-20]
print(x_n.shape, x_h.shape, y_u.shape)          # (928, 20, 14) (928, 20, 14) (928,)

xn_train, xn_test, xh_train, xh_test, yu_train, yu_test = train_test_split(x_n, x_h, y_u, train_size=0.8, random_state=123)
# print(xn_train.shape, xn_test.shape, xh_train.shape, xh_test.shape, yu_train.shape, yu_test.shape)

#2-1. 모델 구성1
input1 = Input(shape=(20,6))
dense11 = Conv1D(32, kernel_size=3, activation='relu', strides=1, padding='same')(input1)
Pooling1 = MaxPooling1D()(dense11)
# drop11 = Dropout(0.3)(Pooling1)
dense12 = Conv1D(64, kernel_size=3, activation='relu', strides=1, padding='same')(Pooling1)
Pooling2 = MaxPooling1D()(dense12)
# drop12 = Dropout(0.2)(Pooling2)
flatten = (Flatten())(Pooling2)
drop13 = Dropout(0.5)(flatten)
dense13 = Dense(128, activation='relu')(drop13)
dense14 = Dense(64, activation='relu')(dense13)
output1 = Dense(32, activation='relu')(dense14)

#2-2. 모델 구성2
input2 = Input(shape=(20,6))
dense21 = LSTM(32, activation='relu', return_sequences=True)(input2)
drop21 = Dropout(0.3)(dense21)
dense22 = LSTM(64, activation='relu')(drop21)
drop22 = Dropout(0.2)(dense22)
dense23 = Dense(128, activation='relu')(drop22)
dense24 = Dense(64, activation='relu')(dense23)
output2 = Dense(32, activation='relu')(dense24)

#2-4. 모델 결합
merge1 = Concatenate(name='mg1')([output1, output2])     # concatenate과 동일
merge2 = Dense(32, name='mg2')(merge1)
merge3 = Dense(16, name='mg3')(merge2)
last_output = Dense(1, name='last')(merge3)
model = Model(inputs=[input1, input2], outputs=last_output)

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

es = EarlyStopping(monitor='val_loss', mode='min', patience=20, verbose=1, restore_best_weights=True,)

########## mcp 세이브 파일명 만들기 시작 ##########
import datetime
date = datetime.datetime.now()
print(date)
print(type(date))

date = date.strftime("%m%d.%H%M")
print(date)
print(type(date))

path = './_save/keras63/'
filename = '{epoch:04d}_valloss_{val_loss:.4f}.hdf5'
filepath = "".join([path, 'k63_01_date_', date, '_epo_', filename])

########## mcp 세이브 파일명 만들기 끝 ##########

mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1, save_best_only=True, filepath=filepath)

start = time.time()
model.fit([xn_train, xh_train], yu_train, epochs=1000, batch_size=16, validation_split=0.2, verbose=3, callbacks=[es, mcp])
end = time.time()

#4. 평가, 예측
loss = model.evaluate([xn_test, xh_test], yu_test)
print("loss :", loss)

y_predict = model.predict([xn_predict, xh_predict])
print('예측값 : ', y_predict)

print('loss :',loss, '(',round(end-start,2),'초) /', y_predict)

'''
random_state=123
loss : 416897.125 (513.69 초) / [[7596.432]] > k63_01_date_0816.1646_epo_0354_valloss_349296.9062

'''
