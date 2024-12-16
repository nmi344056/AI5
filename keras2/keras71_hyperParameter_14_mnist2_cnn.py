import numpy as np
import pandas as pd
import time
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from sklearn.metrics import r2_score, accuracy_score

#1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train.shape, y_train.shape)     # (60000, 28, 28) (60000,)  뒤에 1은 생략, reshape해서 (60000, 28, 28, 1)로 변경
print(x_test.shape, y_test.shape)       # (10000, 28, 28) (10000,)

########## 스케일링 1-1 ##########
x_train = x_train/255.
x_test = x_test/255.
print(np.max(x_train), np.min(x_train))     # 1.0 0.0

x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(10000, 28, 28, 1)
print(x_train.shape, x_test.shape)      # (60000, 28, 28, 1) (10000, 28, 28, 1)

########## OneHotEncoding 3 ##########
from sklearn.preprocessing import OneHotEncoder
y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)
print(y_train.shape, y_test.shape)          # (60000, 1) (10000, 1)

ohe = OneHotEncoder(sparse=False)           # True가 default
y_train = ohe.fit_transform(y_train)
y_test = ohe.fit_transform(y_test)
print(y_train.shape, y_test.shape)          # (60000, 10) (10000, 10)

# model = Sequential()
# model.add(Conv2D(64, (3,3), activation='relu', input_shape=(28, 28, 1)))   # (26, 26, 64)
# model.add(Conv2D(filters=64, kernel_size=(3,3), activation='relu'))        # (24, 24, 64)
# model.add(Dropout(0.2))
# model.add(Conv2D(32, (2,2), activation='relu'))                            # (23, 23, 32)
# model.add(Flatten())                                    # 23*23*32

# model.add(Dense(units=32, activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(units=16, input_shape=(32,), activation='relu'))
# model.add(Dense(10, activation='softmax'))

#2. 모델 구성
def build_model(drop=0.5, optimizer=Adam(0.01), activation='relu',
                node1=128, node2=64, node3=32, node4=16, node5=8,
                ks1=2, ks2=3, lr=0.001):
    inputs = Input(shape=(28, 28, 1), name='inputs')
    x = Conv2D(node1, ks1, activation=activation, name='hidden1')(inputs)
    x = Dropout(drop)(x)
    x = Conv2D(node2, ks2, activation=activation, name='hidden2')(x)
    x = Dropout(drop)(x)
    x = Dense(node3, ks1, activation=activation, name='hidden3')(x)
    x = Flatten()(x)
    x = Dense(node4, activation=activation, name='hidden4')(x)
    x = Dense(node5, activation=activation, name='hidden5')(x)
    outputs = Dense(10, activation='softmax', name='outputs')(x)

    model = Model(inputs=inputs, outputs=outputs)

    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    
    return model

def create_hyperparameter():
    batchs = [32, 16, 8, 1, 64]
    optimizers = ['Adam', 'rmsprop', 'adadelta']
    # lr = [0.01, 0.005, 0.001, 0.0005]
    dropouts = [0.2, 0.3, 0.4, 0.5]
    activations = ['relu', 'elu', 'selu', 'linear']
    node1 = [128, 64, 32, 16]
    node2 = [128, 64, 32, 16]
    node3 = [128, 64, 32, 16]
    node4 = [128, 64, 32, 16]
    node5 = [128, 64, 32, 16, 8]
    return {'batch_size' : batchs,
            'optimizer' : optimizers,
            # 'lr' : lr,
            'drop' : dropouts,
            'activation' : activations,
            'node1' : node1,
            'node2' : node2,
            'node3' : node3,
            'node4' : node4,
            'node5' : node5,
            }

hyperparameters = create_hyperparameter()
print(hyperparameters)
# a

from sklearn.model_selection import RandomizedSearchCV

# sklearn.utils._param_validation.InvalidParameterError: The 'estimator' parameter of RandomizedSearchCV must be an object implementing 'fit'. Got <function build_model at 0x0000022B3EE6FA60> instead.
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

Keras_model = KerasClassifier(build_fn=build_model, verbose=1)

model = RandomizedSearchCV(Keras_model, hyperparameters, cv=3,
                           n_iter=2,
                        #    n_jobs=-1,
                           verbose=1,
                           )

########## mcp 세이브 파일명 만들기 시작 ##########
import datetime
date = datetime.datetime.now()
date = date.strftime("%m%d_%H%M")

path_w = './_save/keras71/'
filename = '{epoch:04d}_valloss_{val_loss:.4f}.hdf5'
filepath = "".join([path_w, 'k71_14_date_', date, '_epo_', filename])

########## mcp 세이브 파일명 만들기 끝 ##########

es = EarlyStopping(
    monitor = 'val_loss',
    mode = 'min',
    patience = 20,
    restore_best_weights = True
)

rlr = ReduceLROnPlateau(
    monitor = 'val_loss',
    mode = 'auto',
    patience = 10,
    verbose = 1,
    factor = 0.8
)

mcp = ModelCheckpoint(
    monitor = 'val_loss',
    mode = 'auto',
    verbose = 1,
    save_best_only = True,
    filepath = filepath
)

import time
start = time.time()
model.fit(x_train, y_train, epochs=5, validation_split = 0.25, callbacks = [es, mcp, rlr], verbose=0)
end = time.time()

print('model.best_params_ :', model.best_params_)
print('model.best_estimator_ :', model.best_estimator_)
print('model.best_score_ :', model.best_score_)
print('model.score :', model.score(x_test, y_test))
print('time :', round(end - start, 2))

from sklearn.metrics import accuracy_score, r2_score
y_predict = model.predict(x_test)
print('acc :',accuracy_score(y_test, y_predict))

'''


'''
