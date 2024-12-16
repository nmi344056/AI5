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
train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    vertical_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1,
    rotation_range=5,
    zoom_range=1.2,
    shear_range=0.7,
    fill_mode='nearest',
)

test_datagen = ImageDataGenerator(rescale=1./255,)

path_train = 'C:\\ai5\\_data\\kaggle\\dogs-vs-cats-redux-kernels-edition\\train\\'
path_test2 = 'C:\\ai5\\_data\\kaggle\\dogs-vs-cats-redux-kernels-edition\\test2\\'
path = 'C:\\ai5\\_data\\kaggle\\dogs-vs-cats-redux-kernels-edition\\'

mission = pd.read_csv(path + "sample_submission.csv", index_col=0)
print(mission)      # [12500 rows x 1 columns]

start1 = time.time()

xy_train = train_datagen.flow_from_directory(
    path_train,
    target_size=(100,100),  # resize, 동일한 규격 사용
    batch_size=25000,
    class_mode='binary',
    color_mode='rgb',
    shuffle=True,
)   # Found 25000 images belonging to 2 classes.

xy_test = test_datagen.flow_from_directory(
    path_test2,
    target_size=(100,100),  # resize, 동일한 규격 사용
    batch_size=12500,
    class_mode='binary',
    color_mode='rgb',
    shuffle=False, 
)   # Found 

x = xy_train[0][0]
y = xy_train[0][1]

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, stratify = y, random_state=5321)

end1 = time.time()
print("time :", round(end1 - start1, 2),'초')    # time : 46.8 초

print(x_train.shape, y_train.shape)         # (20000, 80, 80, 3) (20000,)
print(x_test.shape, y_test.shape)           # (5000, 80, 80, 3) (5000,)

xy_test = xy_test[0][0]
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
# print(y_predict)            # float 형
# print(y_predict.shape)      # (10000, 10)

# acc = accuracy_score(y_test, y_predict)
# print('accuracy_score :', acc)
# print("time :", round(end2 - start2, 2),'초')

y_submit = model.predict(xy_test, batch_size=100)
print(y_submit)
print(y_submit.shape)

mission['label'] = y_submit
print(mission)           # [715 rows x 1 columns]
print(mission.shape)     # (715, 1)

mission.to_csv(path + "teacher_0805.csv")

'''


'''
