import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

# [실습] outliers 확인 및 처리

#1. 데이터
path = 'C:\\ai5\\_data\\kaggle\\wine\\'
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)

def detect_outliers(data):
    for col in data.columns:
        print(f"=== {col} ===")
        col_data = data[col]
        quartile_1, q2, quartile_3 = np.percentile(col_data, [25, 50, 75])
        
        print("1사분위:", quartile_1)
        print("중앙값(q2):", q2)
        print("3사분위:", quartile_3)
        
        iqr = quartile_3 - quartile_1
        print("IQR:", iqr)
        
        low_bound = quartile_1 - (iqr * 1.5)
        upper_bound = quartile_3 + (iqr * 1.5)
        
        print("하한선:", low_bound)
        print("상한선:", upper_bound)
        
        # 이상치 인덱스 출력
        outliers = col_data[(col_data < low_bound) | (col_data > upper_bound)]
        print("이상치:")
        print(outliers)
        
        # 박스플롯 시각화
        plt.figure(figsize=(6, 4))
        plt.boxplot(col_data, vert=False)
        plt.axvline(low_bound, color='red', linestyle='--', label="Lower Bound")
        plt.axvline(upper_bound, color='blue', linestyle='--', label="Upper Bound")
        plt.title(f"Boxplot of {col}")
        plt.legend()
        plt.show()

detect_outliers(train_csv)
