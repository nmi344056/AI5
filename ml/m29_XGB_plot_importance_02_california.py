# 23_1 copy

from sklearn.datasets import fetch_california_housing
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor

# 1. 데이터
datasets = fetch_california_housing()      # feature_name 때문에
x = datasets.data
y = datasets.target

random_state1=1223
random_state2=1223

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, random_state=random_state1)

#2. 모델 구성
# model1 = DecisionTreeRegressor(random_state=random_state2)
# model2 = RandomForestRegressor(random_state=random_state2)
# model3 = GradientBoostingRegressor(random_state=random_state2)
model = XGBRegressor(random_state=random_state2)

import matplotlib.pyplot as plt
import numpy as np

# print(model)

# def plot_feature_importances_dataset(model):
#     n_features = datasets.data.shape[1]
#     plt.barh(np.arange(n_features), model.feature_importances_,align='center')
#     plt.yticks(np.arange(n_features), datasets.feature_names)
#     plt.xlabel('Feature Importances')
#     plt.ylabel('Features')
#     plt.ylim(-1, n_features)
#     plt.title(model.__class__.__name__)

print('random_state :', random_state1, random_state2)

model.fit(x_train, y_train)
print('==========', model.__class__.__name__, '==========')
print('acc :', model.score(x_test, y_test))
print(model.feature_importances_)

from xgboost.plotting import plot_importance
plot_importance(model) 
plt.show()

'''
========== XGBRegressor ==========
acc : 0.8384930657222394
[0.49375907 0.06520814 0.04559402 0.02538511 0.02146595 0.14413244
 0.0975963  0.10685894]
'''
