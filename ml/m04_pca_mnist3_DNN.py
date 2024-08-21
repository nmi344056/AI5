'''
[실습] m04_1에서 뽑은 4가지 결과로 4가지 모델 만들기
input_shape = ()
1.

시간과 성능 체크
결과1
PCA :
걸린시간 :
acc :
'''

import numpy as np
import time
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, Input, MaxPooling2D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, accuracy_score

(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(x_train.shape, x_test.shape)      # (60000, 28, 28) (10000, 28, 28)

# x = np.concatenate([x_train, x_test], axis=0)
# print(x.shape)                        # (70000, 28, 28)

# 스케일링
x_train = x_train/255.
x_test = x_test/255.

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1] * x_train.shape[2])
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1] * x_test.shape[2])
print(x_train.shape, x_test.shape)      # (60000, 784) (10000, 784)

from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# pca = PCA(n_components=28*28)
# x = pca.fit_transform(x)

# evr_cumsum = np.cumsum(pca.explained_variance_ratio_)
# print(evr_cumsum)

# print('0.95 :', np.argmax(evr_cumsum>=0.95)+1)      # 154
# print('0.99 :', np.argmax(evr_cumsum>=0.99)+1)      # 331
# print('0.999 :', np.argmax(evr_cumsum>=0.999)+1)    # 486
# print('1.0 :', np.argmax(evr_cumsum>=1.0)+1)        # 713

for i in range(4):
    num = [154, 331, 486, 713]
    pca = PCA(n_components=num[i])
    x_train1 = pca.fit_transform(x_train)
    x_test1 = pca.transform(x_test)

    #2. 모델 구성
    model = Sequential()
    model.add(Dense(128, activation='relu', input_shape=(num[i],)))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(10, activation='softmax'))

    #3. 컴파일, 훈련
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    es = EarlyStopping(monitor='val_loss', mode='min', patience=10, verbose=0, restore_best_weights=True,)

    ########## mcp 세이브 파일명 만들기 시작 ##########
    import datetime
    date = datetime.datetime.now()
    # print(date)
    # print(type(date))

    date = date.strftime("%m%d_%H%M")
    # print(date)
    # print(type(date))

    path = './_save/m04/'
    filename = '{epoch:04d}_valloss_{val_loss:.4f}.hdf5'
    filepath = "".join([path, 'm04_03_date_', str(i+1), '_', date, '_epo_', filename])

    ########## mcp 세이브 파일명 만들기 끝 ##########

    mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=0, save_best_only=True, filepath=filepath)

    start = time.time()
    hist = model.fit(x_train1, y_train, epochs=1, batch_size=16, validation_split=0.2, verbose=0, callbacks=[es, mcp])
    end = time.time()

    #4. 평가, 예측
    loss = model.evaluate(x_test1, y_test, verbose=0)
    # print("loss : ", loss[0])
    # print("accuracy : ", round(loss[1], 3))

    y_predict = model.predict(x_test1, verbose=0)
    # print(y_predict)            # float 형
    # print(y_predict.shape)      # (10000, 10)

    y_predict = np.argmax(y_predict, axis=1).reshape(-1,1)
    # print(y_predict)            #  int 형
    # print(y_predict.shape)      # (10000, 1)

    y_test1 = np.argmax(y_test, axis=1).reshape(-1,1)
    # print(y_test)
    # print(y_test.shape)

    ##### print #####
    print('결과', i+1)
    print('PCA :', num[i])
    print('time :', round(end-start,2),'초')
    print('accuracy :', round(loss[1], 3))
    print(' ')

'''
결과 1
PCA : 154
time : 1.93 초
accuracy : 0.954

결과 2
PCA : 331
time : 1.38 초
accuracy : 0.951

결과 3
PCA : 486
time : 2.17 초
accuracy : 0.948

결과 4
PCA : 713
time : 1.43 초
accuracy : 0.955

'''
