from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from sklearn.metrics import r2_score, accuracy_score
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.datasets import load_digits
import numpy as np
import pandas as pd
import time

#1. 데이터
x, y = load_digits(return_X_y=True)     #로 분할 가능
print(x)                    # [[]...[]]
print(y)                    # [0 1 2 ... 8 9 8]
print(x.shape, y.shape)     # (1797, 64) (1797,)

print(pd.value_counts(y, sort=False))
# 0    178
# 1    182
# 2    177
# 3    183
# 4    181
# 5    182
# 6    181
# 7    179
# 8    174
# 9    180

print(np.unique(y, return_counts=True))
# (array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]), array([178, 182, 177, 183, 181, 182, 181, 179, 174, 180], dtype=int64))

y_ohe1 = to_categorical(y)
print(y_ohe1)
print(y_ohe1.shape)         # (1797, 10)

x_train, x_test, y_train, y_test = train_test_split(x, y_ohe1, train_size=0.9, random_state=6666)

from sklearn.preprocessing import MinMaxScaler, StandardScaler
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

from sklearn.decomposition import PCA
pca = PCA(n_components=len(x[1]))
x_train = pca.fit_transform(x_train)
x_test = pca.transform(x_test)

evr_cumsum = np.cumsum(pca.explained_variance_ratio_)

num = [np.argmax(evr_cumsum>=0.95)+1, np.argmax(evr_cumsum>=0.99)+1, np.argmax(evr_cumsum>=0.999)+1, np.argmax(evr_cumsum)+1]
results = []

for i in range(0, len(num), 1):
    pca = PCA(n_components=num[i])
    x_train1 = pca.fit_transform(x_train)
    x_test1 = pca.transform(x_test)

    #2. 모델구성
    model = Sequential()
    model.add(Dense(128, activation='relu', input_shape=(num[i],)))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(10, activation='softmax'))

    #3. 컴파일, 훈련
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    es = EarlyStopping(monitor='val_loss', mode='min', patience=50, verbose=1, restore_best_weights=True,)

    ########## mcp 세이브 파일명 만들기 시작 ##########
    import datetime
    date = datetime.datetime.now()
    date = date.strftime("%m%d_%H%M")

    path = './_save/m05/'
    filename = '{epoch:04d}_valloss_{val_loss:.4f}.hdf5'
    filepath = "".join([path, 'm05_11_date_', str(i+1), '_', date, '_epo_', filename])

    ########## mcp 세이브 파일명 만들기 끝 ##########

    mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1, save_best_only=True, filepath=filepath)

    start = time.time()
    hist = model.fit(x_train1, y_train, epochs=100, batch_size=100, validation_split=0.2, callbacks=[es, mcp])
    end = time.time()

    #4. 평가, 예측
    loss = model.evaluate(x_test1, y_test, verbose=1)
    # print("loss : ", loss[0])
    # print("accuracy : ", round(loss[1], 3))

    y_predict = model.predict(x_test1)
    # print(y_predict[:20])
    y_predict1 = np.round(y_predict)
    # print(y_predict[:20])

    results.append([i+1, num[i], round(end-start,2), round(loss[1], 3)])

for a in results:
    print('결과', a[0])
    print('PCA :', a[1])
    print('time :', a[2],'초')
    print('accuracy :', a[3])
    print('')

'''
결과 1
PCA : 30
time : 3.67 초
accuracy : 0.983

결과 2
PCA : 44
time : 3.04 초
accuracy : 0.961

결과 3
PCA : 55
time : 3.98 초
accuracy : 0.961

결과 4
PCA : 61
time : 4.37 초
accuracy : 0.983
'''
