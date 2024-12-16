# [실습] train_test_split 하고 스케일링 후 PCA

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA

#1. 데이터
datasets = load_iris()
x = datasets['data']
y = datasets.target
print(x.shape, y.shape)     # (150, 4) (150,)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=6543, stratify=y)

##### 보통 스케일링 후 PCA #####
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

pca = PCA(n_components=3)
x_train = pca.fit_transform(x_train)
x_test = pca.transform(x_test)

print(x_train, x_train.shape)

#2. 모델 구성
model = RandomForestClassifier(random_state=123)

#3. 훈련
model.fit(x_train, y_train)

#4. 평가, 예측
results = model.score(x_test, y_test)
print(x.shape, '> model.score :', results)

'''
(150, 4) > model.score : 0.9333333333333333
(150, 3) > model.score : 0.9333333333333333
(150, 2) > model.score : 0.9333333333333333
(150, 1) > model.score : 0.9333333333333333

model.score : 1.0
train_size=0.8, random_state=3456 > 2
train_size=0.8, random_state=6543 > 4

'''

evr = pca.explained_variance_ratio_     # 설명가능한 변화율
print(evr, '/', sum(evr))
'''
4 > [0.72807981 0.22870507 0.03809122 0.00512389] / 1.0         # 합치면 1
3 > [0.72807981 0.22870507 0.03809122] / 0.9948761096944939     # 손실이 있었다
2 > [0.72807981 0.22870507] / 0.9567848868935855
1 > [0.72807981]

'''

evr_cumsum = np.cumsum(evr)
print(evr_cumsum)           # [0.72807981 0.95678489 0.99487611 1.        ]

import matplotlib.pyplot as plt
plt.plot(evr_cumsum)
plt.grid()
plt.show()
