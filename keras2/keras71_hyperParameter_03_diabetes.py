import numpy as py
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input

import warnings
warnings.filterwarnings('ignore')

#1. 데이터
x, y = load_diabetes(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=336)

print(x_train.shape, y_train.shape)     # (353, 10) (353,)

#2. 모델 구성
def build_model(drop=0.5, optimizer='adam', activation='relu',
                node1=128, node2=64, node3=32, node4=16, node5=8, lr=0.001):
    inputs = Input(shape=(10, ), name='inputs')
    x = Dense(node1, activation=activation, name='hidden1')(inputs)
    x = Dropout(drop)(x)
    x = Dense(node2, activation=activation, name='hidden2')(x)
    x = Dropout(drop)(x)
    x = Dense(node3, activation=activation, name='hidden3')(x)
    x = Dropout(drop)(x)
    x = Dense(node4, activation=activation, name='hidden4')(x)
    x = Dense(node5, activation=activation, name='hidden5')(x)
    outputs = Dense(1, activation='linear', name='outputs')(x)

    model = Model(inputs=inputs, outputs=outputs)

    model.compile(loss='mse', optimizer=optimizer, metrics=['mae'])
    
    return model

def create_hyperparameter():
    batchs = [32, 16, 8, 1, 64]
    optimizers = ['adam', 'rmsprop', 'adadelta']
    dropouts = [0.2, 0.3, 0.4, 0.5]
    activations = ['relu', 'elu', 'selu', 'linear']
    node1 = [128, 64, 32, 16]
    node2 = [128, 64, 32, 16]
    node3 = [128, 64, 32, 16]
    node4 = [128, 64, 32, 16]
    node5 = [128, 64, 32, 16, 8]
    return {'batch_size' : batchs,
            'optimizer' : optimizers,
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
# {'batch_size': [100, 200, 300, 400, 500], 'optimizer': ['adam', 'rmsprop', 'adadelta'], 'drop': [0.2, 0.3, 0.4, 0.5], 'activation': ['relu', 'elu', 'selu', 'linear'], 
# 'node1': [128, 64, 32, 16], 'node2': [128, 64, 32, 16], 'node3': [128, 64, 32, 16], 'node4': [128, 64, 32, 16], 'node5': [128, 64, 32, 16, 8]}

from sklearn.model_selection import RandomizedSearchCV

# sklearn.utils._param_validation.InvalidParameterError: The 'estimator' parameter of RandomizedSearchCV must be an object implementing 'fit'. Got <function build_model at 0x0000022B3EE6FA60> instead.
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor

Keras_model = KerasRegressor(build_fn=build_model, verbose=1)

model = RandomizedSearchCV(Keras_model, hyperparameters, cv=5,
                           n_iter=10,
                        #    n_jobs=-1,
                           verbose=1,
                           )

import time
start = time.time()
model.fit(x_train, y_train, epochs=100, verbose=0)
end = time.time()

print('model.best_params_ :', model.best_params_)
print('model.best_estimator_ :', model.best_estimator_)
print('model.best_score_ :', model.best_score_)
print('model.score :', model.score(x_test, y_test))
print('time :', round(end - start, 2))

from sklearn.metrics import accuracy_score, r2_score
y_predict = model.predict(x_test)
print('r2 :',r2_score(y_test, y_predict))

'''
batchs = [100, 200, 300, 400, 500] / epochs=30
model.best_params_ : {'optimizer': 'rmsprop', 'node5': 128, 'node4': 128, 'node3': 16, 'node2': 16, 'node1': 32, 'drop': 0.3, 'batch_size': 200, 'activation': 'linear'}
model.best_estimator_ : <keras.wrappers.scikit_learn.KerasRegressor object at 0x000002D9CD7385E0>
model.best_score_ : -28549.236328125
model.score : -28295.736328125
time : 20.23

batchs = [100, 200, 300, 400, 500] / epochs=100
model.best_params_ : {'optimizer': 'adam', 'node5': 64, 'node4': 16, 'node3': 16, 'node2': 128, 'node1': 128, 'drop': 0.2, 'batch_size': 100, 'activation': 'linear'}
model.best_estimator_ : <keras.wrappers.scikit_learn.KerasRegressor object at 0x000001618CB985B0>
model.best_score_ : -3052.607861328125
model.score : -3001.3525390625
time : 56.61

batchs = [32, 16, 8, 1, 64] / epochs=100
model.best_params_ : {'optimizer': 'rmsprop', 'node5': 8, 'node4': 16, 'node3': 128, 'node2': 16, 'node1': 16, 'drop': 0.2, 'batch_size': 1, 'activation': 'elu'}
model.best_estimator_ : <keras.wrappers.scikit_learn.KerasRegressor object at 0x000001EC67A79670>
model.best_score_ : -3050.115625
model.score : -3009.043212890625
time : 2264.6

'''
