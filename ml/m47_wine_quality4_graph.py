'''
[실습] 그래프를 그린다.
1. value_count 사용 X
2. np.unique의 return_count 사용 X
3. groupby 사용, count() 사용       *****

quality를 plt.bar로 그린다.
hint : 데이터의 개수(y축) = 데이터 개수 ...

'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 데이터 로드
path = 'C:\\ai5\\_data\\kaggle\\wine\\'
train_csv = pd.read_csv(path + 'train.csv', index_col=0)

# quality를 groupby()와 count()로 계산하고 bar plot으로 시각화
def plot_quality_distribution(data):
    quality_counts = data.groupby('quality').size()
    
    # Bar plot 시각화
    plt.figure(figsize=(8, 6))
    plt.bar(quality_counts.index, quality_counts.values, color='skyblue')
    plt.title("Wine Quality Distribution")
    plt.xlabel("Quality")
    plt.ylabel("Count")
    plt.xticks(quality_counts.index)  # Ensure quality labels are on the x-axis
    plt.show()

plot_quality_distribution(train_csv)
