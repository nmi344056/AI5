import numpy as np
import pandas as pd
import time
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM, GRU
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

학생csv = 'jena_홍길동.csv'

path1 = "C:\\ai5\\_data\\kaggle\\jena\\"        # 원본csv 데이터 저장 위치
path2 = 'C:\\ai5\\_save\\keras55\\'             # 가중치 파일과 생성된 csv 파일 저장 위치

datasets = pd.read_csv(path1 + "jena_climate_2009_2016.csv", index_col=0)
print(datasets)
print(datasets.shape)

y_정답 = datasets.iloc[-144:, 1]
print(y_정답)
print(y_정답.shape)

학생꺼  = pd.read_csv(path2 + 학생csv, index_col=0)
print(학생꺼)

print(y_정답[:5])
print(학생꺼[:5])

def RMSE(y_test,y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
rmse = RMSE(y_정답, 학생꺼)
print('RMSE : ', rmse)          # 결과 : RMSE :  1.163784915583028

# 스케일링 추가했으면 코드 추가 필요
