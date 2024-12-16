import numpy as np
import pandas as pd
import time
from tensorflow.keras.datasets import mnist, fashion_mnist, cifar100
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from sklearn.metrics import r2_score, accuracy_score

#1. 데이터
(x_train, y_train), (x_test, y_test) = cifar100.load_data()

print(x_train.shape, y_train.shape)     # (50000, 32, 32, 3) (50000, 1) 컬러 데이터
print(x_test.shape, y_test.shape)       # (10000, 32, 32, 3) (10000, 1)

# print(np.unique(y_train, return_counts=True))

x_train = x_train/255.
x_test = x_test/255.
print(np.max(x_train), np.min(x_train)) # 1.0 0.0

x_train = x_train.reshape(50000, 32*32*3)
x_test = x_test.reshape(10000, 32*32*3)
print(x_train.shape, x_test.shape)

from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(sparse=False)
y_train = ohe.fit_transform(y_train)
y_test = ohe.fit_transform(y_test)
print(y_train.shape, y_test.shape)  # (50000, 100) (10000, 100)

from sklearn.decomposition import PCA
pca = PCA(n_components=len(x_train[1]))
x_train = pca.fit_transform(x_train)
x_test = pca.transform(x_test)

evr_cumsum = np.cumsum(pca.explained_variance_ratio_)

num = [np.argmax(evr_cumsum>=0.95)+1, np.argmax(evr_cumsum>=0.99)+1, np.argmax(evr_cumsum>=0.999)+1, np.argmax(evr_cumsum)+1]
results = []

for i in range(len(num)):
    pca = PCA(n_components=num[i])
    x_train1 = pca.fit_transform(x_train)
    x_test1 = pca.transform(x_test)

    #2. 모델 구성
    model = Sequential()
    model.add(Dense(128, activation='relu', input_shape=(num[i],)))   # (26, 26, 64)
    model.add(Dense(64, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(100, activation='softmax'))

    #3. 컴파일, 훈련
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    es = EarlyStopping(monitor='val_loss', mode='min', patience=20, verbose=1, restore_best_weights=True,)

    ########## mcp 세이브 파일명 만들기 시작 ##########
    import datetime
    date = datetime.datetime.now()
    date = date.strftime("%m%d_%H%M")

    path = './_save/m05/'
    filename = '{epoch:04d}_valloss_{val_loss:.4f}.hdf5'
    filepath = "".join([path, 'm05_17_date_', str(i+1), '_', date, '_epo_', filename])

    ########## mcp 세이브 파일명 만들기 끝 ##########

    mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1, save_best_only=True, filepath=filepath)

    start = time.time()
    hist = model.fit(x_train1, y_train, epochs=500, batch_size=256, validation_split=0.2, callbacks=[es, mcp])
    end = time.time()

    #4. 평가, 예측
    loss = model.evaluate(x_test1, y_test, verbose=1)
    print("loss : ", loss[0])
    print("accuracy : ", round(loss[1], 3))

    y_predict = model.predict(x_test1)
    # print(y_predict)            # float 형
    # print(y_predict.shape)      # (10000, 100)

    # y_predict = np.argmax(y_predict, axis=1).reshape(-1,1)
    # print(y_predict)            #  int 형
    # print(y_predict.shape)      # (10000, 1)

    # y_test = np.argmax(y_test, axis=1).reshape(-1,1)
    # print(y_test)
    # print(y_test.shape)

    # acc = accuracy_score(y_test, y_predict)
    # print('accuracy_score :', acc)
    # print("time :", round(end-start,2),'초')

    print('결과', i+1)
    print('PCA :', num[i])
    print('time :', round(end-start,2),'초')
    print('loss :', round(loss[0], 3))
    print('accuracy :', round(loss[1], 3))
    print('')

#     results.append([i+1, num[i], round(end-start,2), round(loss[0], 3), round(loss[1], 3)])

# for a in results:
#     print('결과', a[0])
#     print('PCA :', a[1])
#     print('time :', a[2],'초')
#     print('loss :', a[3])
#     print('accuracy :', a[4])
#     print('')

'''
결과 1
PCA : 202
time : 1.43 초
loss : 4.145
accuracy : 0.072

결과 2
PCA : 659
time : 1.32 초
loss : 4.131
accuracy : 0.073

결과 3
PCA : 1481
time : 1.6 초
loss : 4.183
accuracy : 0.052

결과 4
PCA : 3072
time : 1.97 초
loss : 4.143
accuracy : 0.063

'''
