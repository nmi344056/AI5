import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM, GRU
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split

a = np.array(range(1, 101))
x_predict = np.array(range(96, 106))

# [실습] 101부터 107을 찾아라

def split_x(dataset, size):
    aaa = []
    for i in range(len(dataset) - size + 1):
        subset = dataset[i : (i + size)]
        aaa.append(subset)
    return np.array(aaa)

bbb = split_x(a, 6)
print(bbb)
'''
[...
 [ 86  87  88  89  90  91]
 [ 87  88  89  90  91  92]
 [ 88  89  90  91  92  93]
 [ 89  90  91  92  93  94]
 [ 90  91  92  93  94  95]
 [ 91  92  93  94  95  96]
 [ 92  93  94  95  96  97]
 [ 93  94  95  96  97  98]
 [ 94  95  96  97  98  99]
 [ 95  96  97  98  99 100]]
'''
print(bbb.shape)        # (95, 6)

bbb = split_x(bbb, 7)
print(bbb)
'''
[...
 [[ 88  89  90  91  92  93]
  [ 89  90  91  92  93  94]
  [ 90  91  92  93  94  95]
  [ 91  92  93  94  95  96]
  [ 92  93  94  95  96  97]
  [ 93  94  95  96  97  98]
  [ 94  95  96  97  98  99]]
  
 [[ 89  90  91  92  93  94]
  [ 90  91  92  93  94  95]
  [ 91  92  93  94  95  96]
  [ 92  93  94  95  96  97]
  [ 93  94  95  96  97  98]
  [ 94  95  96  97  98  99]
  [ 95  96  97  98  99 100]]]
'''

print(bbb.shape)        # (89, 7, 6)

x = bbb[:, :-1, :-1]
y = bbb[:, :, -1]

# print(x)
'''
[...
 [[89 90 91 92 93]
  [90 91 92 93 94]
  [91 92 93 94 95]
  [92 93 94 95 96]
  [93 94 95 96 97]
  [94 95 96 97 98]]]
'''
# print(y)
# [... [ 94  95  96  97  98  99 100]]

print(x.shape, y.shape)     # (89, 6, 5) (89, 7)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=123)
print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

#2. 모델 구성
model = Sequential()
model.add(LSTM(16, input_shape=(6, 5)))
model.add(Dense(32))
model.add(Dense(64))
model.add(Dense(128))
model.add(Dense(64))
model.add(Dense(32))
model.add(Dense(16))
model.add(Dense(7))

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

es = EarlyStopping(monitor='val_loss', mode='min', patience=20, verbose=1, restore_best_weights=True,)

########## mcp 세이브 파일명 만들기 시작 ##########
import datetime
date = datetime.datetime.now()
print(date)
print(type(date))

date = date.strftime("%m%d.%H%M")
print(date)
print(type(date))

path = './_save/keras54/'
filename = '{epoch:04d}_valloss_{val_loss:.4f}.hdf5'
filepath = "".join([path, 'k54_03_date_', date, '_epo_', filename])

########## mcp 세이브 파일명 만들기 끝 ##########

mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1, save_best_only=True, filepath=filepath)

hist = model.fit(x_train, y_train, epochs=500, batch_size=1, validation_split=0.2, callbacks=[es, mcp])

model.save("./_save/keras54/k54_03.h5")

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

##### a #####
x_predict = split_x(x_predict, 5)

print(x_predict)
'''
[[ 96  97  98  99 100]
 [ 97  98  99 100 101]
 [ 98  99 100 101 102]
 [ 99 100 101 102 103]
 [100 101 102 103 104]
 [101 102 103 104 105]]
'''
print(x_predict.shape)      # (6, 5)

x_predict = x_predict.reshape(1,6,5)
print(x_predict.shape)      # (1, 6, 5)

y_predict = model.predict(x_predict)
print(y_predict.shape)      # (1, 7)

print('loss : ', loss)
print('예상 결과 101 ~ 107 : ', y_predict)

'''
loss :  82.77296447753906
예상 결과 101 ~ 107 :  [[103.41338  104.69086  105.40957  106.51701  107.568954 108.94859  110.22643 ]]

k54_03_date_0809.1237_epo_0038_valloss_0.7393.hdf5
loss :  [0.7603287696838379, 1.0]
예상 결과 101 ~ 107 :  [[ 95.494965  97.08002   97.83761   98.8526    99.1809   100.77138  102.177155]]

'''
