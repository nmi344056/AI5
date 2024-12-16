import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM, GRU
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# a = np.array([[1,2,3,4,5,6,7,8,9,10],
#               [9,8,7,6,5,4,3,2,1,0]]).reshape(10,2)

a = np.array([[1,2,3,4,5,6,7,8,9,10],
              [9,8,7,6,5,4,3,2,1,0]]).T     # a = a.T

size = 6

# [실습] x=(5, 5, 2), y=(5,) 만들기

# print(a)
'''
[[ 1  9] 
 [ 2  8] 
 [ 3  7] 
 [ 4  6] 
 [ 5  5] 
 [ 6  4] 
 [ 7  3] 
 [ 8  2] 
 [ 9  1] 
 [10  0]]
'''
print(a.shape)  # (10, 2)

def split_x(dataset, size):
    aaa = []
    for i in range(len(dataset) - size + 1):
        subset = dataset[i : (i + size)]
        aaa.append(subset)
    return np.array(aaa)

bbb = split_x(a, size)
print(bbb)
'''
[[[ 1  9] 
  [ 2  8] 
  [ 3  7] 
  [ 4  6] 
  [ 5  5] 
  [ 6  4]]

 [[ 2  8] 
  [ 3  7] 
  [ 4  6] 
  [ 5  5] 
  [ 6  4] 
  [ 7  3]]

 [[ 3  7] 
  [ 4  6] 
  [ 5  5]
  [ 6  4]
  [ 7  3]
  [ 8  2]]

 [[ 4  6]
  [ 5  5]
  [ 6  4]
  [ 7  3]
  [ 8  2]
  [ 9  1]]

 [[ 5  5]
  [ 6  4]
  [ 7  3]
  [ 8  2]
  [ 9  1]
  [10  0]]]
'''
print(bbb.shape)        # (5, 6, 2)

x = bbb[:, :-1]         # bbb[모든행, 모든컬럼에서 제일뒤에한개빼고]
y = bbb[:, -1, 0]
print(x, y)
'''
[[[1 9]
  [2 8]
  [3 7]
  [4 6]
  [5 5]]

 [[2 8]
  [3 7]
  [4 6]
  [5 5]
  [6 4]]

 [[3 7]
  [4 6]
  [5 5]
  [6 4]
  [7 3]]

 [[4 6]
  [5 5]
  [6 4]
  [7 3]
  [8 2]]

 [[5 5]
  [6 4]
  [7 3]
  [8 2]
  [9 1]]] [ 6  7  8  9 10]
'''
print(x.shape, y.shape) # (5, 5, 2) (5,)

#2. 모델 구성
model = Sequential()
model.add(LSTM(16, input_shape=(5,2), return_sequences=True))
model.add(LSTM(32))
model.add(Dense(64))
model.add(Dense(128))
model.add(Dense(64))
model.add(Dense(32))
model.add(Dense(16))
model.add(Dense(1))

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
hist = model.fit(x, y, epochs=100, batch_size=1)

model.save("./_save/keras52/k52_02.h5")

#4. 평가, 예측
results = model.evaluate(x, y)
y_predict = model.predict([[[6,4],[7,3],[8,2],[9,1],[10,0]]])
print('loss : ', results)
print('예측값 11 : ', y_predict)

'''
loss :  0.0004280836437828839
예상 결과 11 :  [[10.77322]]
'''
