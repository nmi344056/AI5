# [실습] 만들어보기
#1. 에서 시간 체크

import numpy as np
import pandas as pd
import time
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score

train_datagen = ImageDataGenerator(rescale=1./255,)
test_datagen = ImageDataGenerator(rescale=1./255,)

path_train = './_data/image/cat_and_dog/Train/'
path_test = './_data/image/cat_and_dog/Test/'

start1 = time.time()

xy_train = train_datagen.flow_from_directory(
    path_train,
    target_size=(100,100),  # resize, 동일한 규격 사용
    batch_size=20000,
    class_mode='binary',
    color_mode='rgb',
    shuffle=True,
)   # Found 19997 images belonging to 2 classes.

xy_test = test_datagen.flow_from_directory(
    path_train,
    target_size=(100,100),  # resize, 동일한 규격 사용
    batch_size=20000,
    class_mode='binary',
    color_mode='rgb',
)   # Found 19997 images belonging to 2 classes.

x_train, x_test, y_train, y_test = train_test_split(xy_train[0][0], xy_train[0][1], train_size=0.8, random_state=123)

print(x_train.shape, y_train.shape)         # (15997, 200, 200, 3) (15997,)
print(x_test.shape, y_test.shape)           # (4000, 200, 200, 3) (4000,)

end1 = time.time()
print("time :", round(end1 - start1, 2),'초')    # time : 46.8 초

#2. 모델 구성
model = Sequential()
model.add(Conv2D(16, (3,3), activation='relu', input_shape=(100, 100, 3)))
model.add(MaxPooling2D())
model.add(Conv2D(filters=16, kernel_size=(3,3), activation='relu'))
model.add(Dropout(0.2))
model.add(Conv2D(8, (2,2), activation='relu'))
model.add(Flatten())

model.add(Dense(units=8, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(units=4, input_shape=(32,), activation='relu'))
model.add(Dense(1, activation='sigmoid'))

#3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

es = EarlyStopping(monitor='val_loss', mode='min', patience=20, verbose=1, restore_best_weights=True,)

########## mcp 세이브 파일명 만들기 시작 ##########
import datetime
date = datetime.datetime.now()
print(date)
print(type(date))

date = date.strftime("%m%d.%H%M")
print(date)
print(type(date))

path = './_save/keras41/'
filename = '{epoch:04d}_valloss_{val_loss:.4f}.hdf5'
filepath = "".join([path, 'k41_03_date_', date, '_epo_', filename])

########## mcp 세이브 파일명 만들기 끝 ##########

mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1, save_best_only=True, filepath=filepath)

start2 = time.time()
hist = model.fit(x_train, y_train, epochs=1000, batch_size=10, validation_split=0.2, callbacks=[es, mcp])
end2 = time.time()

#4. 평가, 예측
loss = model.evaluate(x_test, y_test, verbose=1)
print("loss : ", loss[0])
print("accuracy : ", round(loss[1], 3))

y_predict = model.predict(x_test)
# print(y_predict)            # float 형
# print(y_predict.shape)      # (10000, 10)

# acc = accuracy_score(y_test, y_predict)
# print('accuracy_score :', acc)
print("time :", round(end2 - start2, 2),'초')

'''
loss :  0.5186458230018616
accuracy :  0.745
time : 151.29 초
'''
