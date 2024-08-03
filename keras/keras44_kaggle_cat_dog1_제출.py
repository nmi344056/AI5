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
path_train = 'C:\\ai5\\_data\\kaggle\\dogs-vs-cats-redux-kernels-edition\\train\\'
path_test2 = 'C:\\ai5\\_data\\kaggle\\dogs-vs-cats-redux-kernels-edition\\test2\\'
path = 'C:\\ai5\\_data\\kaggle\\dogs-vs-cats-redux-kernels-edition\\'
mission = pd.read_csv(path + "sample_submission.csv", index_col=0)

start1 = time.time()

np_path = 'C:\\ai5\\_data\\_save_npy\\keras43\\'
# x_train = np.load(np_path + 'keras43_01_x_train.npy')
# y_train = np.load(np_path + 'keras43_01_y_train.npy')
x_test = np.load(np_path + 'keras43_01_x_test.npy')
y_test = np.load(np_path + 'keras43_01_y_test.npy')

# x = x_train
# y = y_train

# x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, stratify = y, random_state=5321)

# end1 = time.time()
# print("time :", round(end1 - start1, 2),'초')    # time : 46.8 초

# print(x_train.shape, y_train.shape)         # (20000, 80, 80, 3) (20000,)
# print(x_test.shape, y_test.shape)           # (5000, 80, 80, 3) (5000,)

# xy_test = x_test
# print(xy_test)
# print(xy_test.shape)

# #2. 모델구성
# model = Sequential()
# model.add(Conv2D(64, (3,3), input_shape=(100, 100, 3), activation='relu', padding='same'))
# model.add(Dropout(0.3))
# model.add(MaxPooling2D(pool_size=(2,2)))
# model.add(Conv2D(128, (3,3), activation='relu', padding='same'))
# model.add(Dropout(0.2))
# model.add(MaxPooling2D(pool_size=(2,2)))
# model.add(Conv2D(256, (3,3), activation='relu', padding='same'))
# model.add(Dropout(0.1))
# model.add(MaxPooling2D(pool_size=(2,2)))
# model.add(Flatten())
# model.add(Dropout(rate=0.5))

# model.add(Dense(256, activation='relu'))
# model.add(Dense(128, activation='relu'))
# model.add(Dense(64, activation='relu'))
# model.add(Dense(32, activation='relu'))
# model.add(Dense(1, activation='sigmoid'))

# #3. 컴파일, 훈련
# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# es = EarlyStopping(monitor='val_loss', mode='min', patience=10, verbose=1, restore_best_weights=True,)
# mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1, save_best_only=True, filepath=filepath)

# start2 = time.time()
# hist = model.fit(x_train, y_train, epochs=1000, batch_size=30, validation_split=0.2, callbacks=[es, mcp])
# end2 = time.time()

#4. 평가, 예측
print('========== 2. mcp 출력 ==========')
path2 = 'C:\\ai5\_data\\kaggle\\dogs-vs-cats-redux-kernels-edition\\w\\'
model = load_model(path2 + 'k42_kaggle_catdog_date_0804.2248_epo_0009_valloss_0.6075.hdf5')

loss = model.evaluate(x_test, y_test, verbose=1)
print("loss : ", loss)

# y_predict = model.predict(x_test, batch_size=100)

y_submit = model.predict(x_test, batch_size=100)        # 변경
print(y_submit)
print(y_submit.shape)

mission['label'] = y_submit
print(mission)           # [715 rows x 1 columns]
print(mission.shape)     # (715, 1)

mission.to_csv(path + "teacher_0805.csv")

'''


'''
