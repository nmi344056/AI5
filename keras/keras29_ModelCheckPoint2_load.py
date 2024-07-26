from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score
from sklearn.datasets import load_boston
import numpy as np
import time

#1. 데이터
dataset = load_boston()

x = dataset.data
y = dataset.target

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=555)

from sklearn.preprocessing import MinMaxScaler, StandardScaler
scaler = MinMaxScaler()

x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# #2. 모델구성
# model = Sequential()
# model.add(Dense(10, input_shape=(13,)))
# model.add(Dense(5))
# model.add(Dense(1))

# #3. 컴파일, 훈련
# model.compile(loss='mse', optimizer='adam')

# from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
# es  = EarlyStopping(monitor='val_loss', mode='min', patience=10, verbose=1, restore_best_weights=True,)
# mcp = ModelCheckpoint(monitor='val_loss', mode='auto',
#                       verbose=1,
#                       save_best_only=True,
#                       filepath='./_save/keras29_mcp/keras29_mcp1.hdf5'
#                       )

# start = time.time()
# hist = model.fit(x_train, y_train, epochs=1000, batch_size=10, validation_split=0.2, callbacks=[es, mcp])
# end = time.time()

model = load_model('./_save/keras29_mcp/keras29_mcp1.hdf5')

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print("loss : ", loss)

y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print("r2 score : ", r2)
# print("time : ", round(end - start, 2), "초")

# '''
# 32 16 16 16 1 / train_size=0.7, random_state=555 / epochs=500, batch_size=10
#                 loss :  23.239059448242188 / r2 score :  0.7301424718220688
# MinMaxScaler > loss :  20.770709991455078 / r2 score :  0.7588055105996
# StandardScaler > loss :  20.400049209594727 / r2 score :  0.7631097044958196 > best
# MaxAbsScaler > loss :  22.515735626220703 / r2 score :  0.7385418668215343
# RobustScaler > loss :  20.481996536254883 / r2 score :  0.7621581380586453 >2
# '''
