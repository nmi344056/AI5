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
from sklearn.metrics import r2_score, accuracy_score

#1. 데이터
path_np = 'C:\\ai5\\_data\\_save_npy\\keras45\\'
x_test = np.load(path_np + 'keras45_01_brain_x_test.npy')
y_test = np.load(path_np + 'keras45_01_brain_y_test.npy')

#2. 모델 구성
# model = Sequential()
# model.add(Conv2D(16, (3,3), activation='relu', input_shape=(200, 200, 1)))
# model.add(MaxPooling2D())
# model.add(Conv2D(filters=16, kernel_size=(3,3), activation='relu'))
# model.add(Dropout(0.2))
# model.add(Conv2D(8, (2,2), activation='relu'))
# model.add(Flatten())

# model.add(Dense(units=8, activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(units=4, input_shape=(32,), activation='relu'))
# model.add(Dense(1, activation='sigmoid'))

# #3. 컴파일, 훈련
# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# es = EarlyStopping(monitor='val_loss', mode='min', patience=20, verbose=1, restore_best_weights=True,)
# mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1, save_best_only=True, filepath=filepath)

# start = time.time()
# hist = model.fit(x_train, y_train, epochs=100, batch_size=160, validation_split=0.2, callbacks=[es, mcp])
# end = time.time()

#4. 평가, 예측
path2 = './_save/keras41/'
model = load_model(path2 + 'k41_02_date_0805.1230_epo_0049_valloss_0.0479.hdf5')

loss = model.evaluate(x_test, y_test, verbose=1)
print("loss : ", loss[0])
print("accuracy : ", round(loss[1], 3))

y_predict = model.predict(x_test)
print(y_predict)            # float 형
print(y_predict.shape)      # (10000, 10)


y_predict = np.round(y_predict)
acc = accuracy_score(y_test, y_predict)
print('accuracy_score :', acc)
# print("time :", round(end-start,2),'초')

'''
[실습] accuracy 0.98 이상
loss :  0.02529209852218628 / accuracy :  1.0 > 동일

'''
