import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
import time
from sklearn.datasets import load_breast_cancer

#1. 데이터
datasets = load_breast_cancer()
print(datasets. DESCR)
print(datasets.feature_names)

x = datasets.data
y = datasets.target
print(x.shape, y.shape)

print(np.unique(y, return_counts=True))
# (array([0, 1]), array([212, 357], dtype=int64))  불균형 데이터인지 확인
print(type(x))  # <class 'numpy.ndarray'>  넘파이 파일

# print(y.value_counts())  # 에러
print(pd.DataFrame(y).value_counts())
print(pd.Series(y).value)
pd.value_counts(y)

x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.9,
                                                    random_state=315)

print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

#2. 모델구성
