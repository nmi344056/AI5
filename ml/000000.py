import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling1D, Bidirectional, LSTM, Conv1D
import time
from sklearn.metrics import r2_score, accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split


#1. 데이터
train_datagen =  ImageDataGenerator(
    # rescale=1./255,              # 이미지를 수치화 할 때 0~1 사이의 값으로 (스케일링 한 데이터로 사용)
    horizontal_flip=True,        # 수평 뒤집기   <- 데이터 증폭 
    vertical_flip=True,          # 수직 뒤집기 (상하좌우반전) <- 데이터 증폭
    width_shift_range=0.2,       # 평행이동  <- 데이터 증폭
    # height_shift_range=0.1,      # 평행이동 수직  <- 데이터 증폭
    rotation_range=15,            # 각도 조절 (정해진 각도만큼 이미지 회전)
    # zoom_range=1.2,              # 축소 또는 확대
    # shear_range=0.7,             # 좌표 하나를 고정시키고 다른 몇개의 좌표를 이동시키는 변환 (찌부시키기)
    fill_mode='nearest',         # 10% 이동 시 한쪽은 소실, 한쪽은 가까이에 있던 부분의 이미지로 채워짐
)

start1 = time.time()
np_path = "c:/ai5/_data/_save_npy/horse/"

x_train = np.load(np_path + 'keras45_02_x_train.npy')
y_train = np.load(np_path + 'keras45_02_y_train.npy')


x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2, random_state=231)
end1 = time.time()

print('데이터 걸린시간 :',round(end1-start1,2),'초')

print(x_train.shape, y_train.shape) # (821, 200, 200, 3) (821,)
print(x_test.shape, y_test.shape)   # (206, 200, 200, 3) (206,)
# 데이터 걸린시간 : 71.87 초


augment_size = 10000  

randidx = np.random.randint(x_train.shape[0], size = augment_size) 
print(randidx)              
print(np.min(randidx), np.max(randidx)) 

print(x_train[0].shape) 

x_augmented = x_train[randidx].copy() 
y_augmented = y_train[randidx].copy()

print(x_augmented.shape)   
print(y_augmented.shape)   

x_augmented = x_augmented.reshape(
    x_augmented.shape[0],      
    x_augmented.shape[1],     
    x_augmented.shape[2], 3)    

print(x_augmented.shape)  

x_augmented = train_datagen.flow(
    x_augmented, y_augmented,
    batch_size=augment_size,
    shuffle=False,
).next()[0]

print(x_augmented.shape)   

x_train = x_train.reshape(821, 200, 200, 3)
x_test = x_test.reshape(206, 200, 200, 3)

print(x_train.shape, x_test.shape) 

# ## numpy에서 데이터 합치기
# x_train = np.concatenate((x_train, x_augmented))
# y_train = np.concatenate((y_train, y_augmented))
print(x_train.shape, y_train.shape)     # (10821, 200, 200, 3) (10821,)

print(np.unique(y_train, return_counts=True))

x_train = x_train.reshape(821,200*200*3)
x_test = x_test.reshape(206,200*200*3)

from sklearn.decomposition import PCA

pca = PCA(n_components=x_train.shape[0])  
x_train = pca.fit_transform(x_train)
x_test = pca.transform(x_test)

evr = pca.explained_variance_ratio_     # 설명가능한 변화율

evr_cumsum = np.cumsum(evr)     #누적합
print(evr_cumsum)

print('0.95 이상 :', np.argmax(evr_cumsum>=0.95)+1)
print('0.99 이상 :', np.argmax(evr_cumsum >= 0.99)+1)
print('0.999 이상 :', np.argmax(evr_cumsum >= 0.999)+1)
print('1.0 일 때 :', np.argmax(evr_cumsum)+1)

# 0.95 이상 : 261
# 0.99 이상 : 564
# 0.999 이상 : 760
# 1.0 일 때 : 820

num = [261, 564, 760, 820]

for i in range(len(num)): 
    pca = PCA(n_components=num[i])   # 4개의 컬럼이 3개로 바뀜
    x_train1 = pca.fit_transform(x_train)
    x_test1 = pca.transform(x_test)
    
    #2. 모델 구성
    model = Sequential()
    model.add(Dense(1024, input_shape=(num[i],)))
    model.add(Dense(1024, activation='relu'))
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    #3. 컴파일, 훈련
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
    es = EarlyStopping(monitor='val_loss', mode='min', 
                    patience=10, verbose=0,
                    restore_best_weights=True,
                    )

    ###### mcp 세이브 파일명 만들기 ######
    import datetime
    date = datetime.datetime.now()
    date = date.strftime("%m%d_%H%M")

    path = './_save/ml05/19_horse/'
    filename = '{epoch:04d}-{val_loss:.4f}.hdf5' 
    filepath = "".join([path, 'ml05_', str(i+1), '_', date, '_', filename])   
    #####################################

    mcp = ModelCheckpoint(
        monitor='val_loss',
        mode='auto',
        verbose=0,     
        save_best_only=True,   
        filepath=filepath, 
    )

    start = time.time()
    hist = model.fit(x_train1, y_train, epochs=1000, batch_size=10,  
            verbose=0, 
            validation_split=0.1,
            callbacks=[es, mcp],
            )
    end = time.time()

    #4. 평가, 예측
    loss = model.evaluate(x_test1, y_test, verbose=0)

    y_pre = model.predict(x_test1, verbose=0)

    r2 = r2_score(y_test, y_pre)  
    
    print('결과', i+1)
    print('PCA :',num[i])
    print('loss :', round(loss[0],8))
    print('acc :', round(loss[1],8))
    print('r2 score :', r2) 
    print("걸린 시간 :", round(end-start,2),'초')
    print("===============================================")

"""
loss : 0.16553156077861786
acc : 0.98544
걸린 시간 : 50.05 초
accuracy_score : 0.9854368932038835

loss : 0.6921750903129578
acc : 0.52913
걸린 시간 : 65.01 초
accuracy_score : 0.529126213592233

[LSTM]
loss : 0.6953322887420654
acc : 0.47087
걸린 시간 : 1991.15 초
accuracy_score : 0.470873786407767

[Conv1D]
loss : 0.6923068165779114
acc : 0.52913
걸린 시간 : 57.01 초
accuracy_score : 0.529126213592233

"""

# 결과 1
# PCA : 261
# loss : 0.51468772
# acc : 0.92718446
# r2 score : 0.7343625321587941
# 걸린 시간 : 26.2 초
# ===============================================
# 결과 2
# PCA : 564
# loss : 0.35302126
# acc : 0.91262138
# r2 score : 0.6976438501483082
# 걸린 시간 : 12.54 초
# ===============================================
# 결과 3
# PCA : 760
# loss : 0.30036265
# acc : 0.94660193
# r2 score : 0.7675825005573826
# 걸린 시간 : 11.97 초
# ===============================================
# 결과 4
# PCA : 820
# loss : 0.15124342
# acc : 0.94174755
# r2 score : 0.820152994415368
# 걸린 시간 : 12.23 초
# ===============================================