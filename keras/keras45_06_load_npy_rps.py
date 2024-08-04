import numpy as np
import pandas as pd
import time
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score

#1. 데이터
start1 = time.time()

np_path = 'C:\\ai5\\_data\\_save_npy\\keras45\\'
x = np.load(np_path + 'keras45_03_rps_x_train.npy')
y = np.load(np_path + 'keras45_03_rps_y_train.npy')

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=123)

end1 = time.time()
print("time :", round(end1 - start1, 2),'초')    # time : 46.8 초

# #2. 모델 구성
# model = Sequential()
# model.add(Conv2D(16, (3,3), activation='relu', input_shape=(100, 100, 3)))
# model.add(MaxPooling2D())
# model.add(Conv2D(filters=16, kernel_size=(3,3), activation='relu'))
# model.add(Dropout(0.2))
# model.add(Conv2D(8, (2,2), activation='relu'))
# model.add(Flatten())

# model.add(Dense(units=8, activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(units=4, input_shape=(32,), activation='relu'))
# model.add(Dense(3, activation='softmax'))

# #3. 컴파일, 훈련
# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# es = EarlyStopping(monitor='val_loss', mode='min', patience=20, verbose=1, restore_best_weights=True,)
# mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1, save_best_only=True, filepath=filepath)

# start2 = time.time()
# hist = model.fit(x_train, y_train, epochs=1000, batch_size=10, validation_split=0.2, callbacks=[es, mcp])
# end2 = time.time()

#4. 평가, 예측
path2 = './_save/keras41/'
model = load_model(path2 + 'k41_05_date_0805.1318_epo_0699_valloss_1.0303.hdf5')

loss = model.evaluate(x_test, y_test, verbose=1)
print("loss : ", loss[0])
print("accuracy : ", round(loss[1], 3))

y_predict = model.predict(x_test)
# print(y_predict)            # float 형
# print(y_predict.shape)      # (10000, 10)

y_predict = np.round(y_predict)
acc = accuracy_score(y_test, y_predict)
print('accuracy_score :', acc)  # 0.0 ? 이상하다.
# print("time :", round(end2 - start2, 2),'초')

'''
loss :  0.8403165340423584
accuracy :  0.667
time : 3.59 초

loss :  1.218637228012085
accuracy :  0.333
time : 3.77 초
'''
