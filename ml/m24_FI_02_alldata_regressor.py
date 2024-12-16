#[실습] cancer, wine, digits

import numpy as np
from sklearn.datasets import fetch_california_housing, load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor

#1. 데이터
datas = [fetch_california_housing, load_diabetes]
name = ['california', 'diabetes']

for i, data in enumerate(datas):
    x, y = data(return_X_y=True)
    # print(x.shape, y.shape)

    random_state1=123
    random_state2=1223

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, train_size=0.8, shuffle=True, random_state=123)

    scaler = MinMaxScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    model1 = DecisionTreeRegressor(random_state=random_state2)
    model2 = RandomForestRegressor(random_state=random_state2)
    model3 = GradientBoostingRegressor(random_state=random_state2)
    model4 = XGBRegressor(random_state=random_state2)

    models = [model1, model2, model3, model4]

    print('##########', name[i], '##########')
    print('random_state :', random_state1, random_state2)
    for model in models:
        model.fit(x_train, y_train)
        print('==========', model.__class__.__name__, '==========')
        print('acc :', model.score(x_test, y_test))
        print(model.feature_importances_)

    print(' ')

'''
########## california ##########
random_state : 123 1223
========== DecisionTreeRegressor ==========
acc : 0.5989423695879894
[0.51983205 0.04856409 0.04957449 0.02667537 0.03118251 0.13285869
 0.09854418 0.09276862]
========== RandomForestRegressor ==========
acc : 0.8128754801930348
[0.52317018 0.05302915 0.04807856 0.02990133 0.03194865 0.13357962
 0.09056457 0.08972796]
========== GradientBoostingRegressor ==========
acc : 0.7949558199558235
[0.59991709 0.02722409 0.02205427 0.00466002 0.00411828 0.12507757
 0.11269092 0.10425775]
========== XGBRegressor ==========
acc : 0.83707103301617
[0.47826383 0.07366086 0.0509511  0.02446287 0.02366972 0.14824368
 0.0921493  0.10859864]

########## diabetes ##########
random_state : 123 1223
========== DecisionTreeRegressor ==========
acc : 0.13480094612764226
[0.09223737 0.01882284 0.22649976 0.05185322 0.0559527  0.04964265
 0.04436473 0.02117035 0.36696253 0.07249386]
========== RandomForestRegressor ==========
acc : 0.5487701498444223
[0.05674201 0.01124774 0.30545693 0.10644086 0.04281331 0.05484868
 0.05792922 0.02794217 0.25241258 0.08416649]
========== GradientBoostingRegressor ==========
acc : 0.5526580735430517
[0.04903104 0.01077472 0.30318419 0.11201041 0.0282878  0.05523065
 0.04114573 0.01782559 0.33841886 0.044091  ]
========== XGBRegressor ==========
acc : 0.39065385219018145
[0.04159961 0.07224615 0.17835377 0.06647415 0.04094251 0.04973729
 0.03822911 0.10475955 0.3368922  0.07076568]
'''
