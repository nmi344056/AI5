import pandas as pd
import numpy as np

data = pd.DataFrame([[2, np.nan, 6, 8, 10],
                     [2, 4, np.nan, 8, np.nan,],
                     [2, 4, 6, 8, 10,],
                     [np.nan, 4, np.nan, 8, np.nan,],
                     ])
# print(data)

data = data.transpose()
data.columns = ['x1', 'x2', 'x3', 'x4']
# print(data)
'''
     x1   x2    x3   x4
0   2.0  2.0   2.0  NaN
1   NaN  4.0   4.0  4.0
2   6.0  NaN   6.0  NaN
3   8.0  8.0   8.0  8.0
4  10.0  NaN  10.0  NaN
'''

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

imputer = IterativeImputer()  # Default : BayesianRidge 회귀모델
                              # interpolate의 상향 버전 : NAN값, 첫값, 마지막값을 찾아낸다
data1 = imputer.fit_transform(data)
print(data1)
'''
[[ 2.          2.          2.          2.0000005 ]
 [ 4.00000099  4.          4.          4.        ]
 [ 6.          5.99999928  6.          5.9999996 ]
 [ 8.          8.          8.          8.        ]
 [10.          9.99999872 10.          9.99999874]]
'''

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

imputer = IterativeImputer(estimator=DecisionTreeRegressor())  # DecisionTreeRegressor로 찾는다
data2 = imputer.fit_transform(data)
print(data2)
'''
[[ 2.  2.  2.  4.]
 [ 6.  4.  4.  4.]
 [ 6.  4.  6.  4.]
 [ 8.  8.  8.  8.]
 [10.  8. 10.  8.]]
'''

imputer = IterativeImputer(estimator=RandomForestRegressor())
data3 = imputer.fit_transform(data)
print(data3)
'''
[[ 2.    2.    2.    4.8 ]
 [ 4.16  4.    4.    4.  ]
 [ 6.    4.04  6.    4.8 ]
 [ 8.    8.    8.    8.  ]
 [10.    6.58 10.    6.88]]
'''

imputer = IterativeImputer(estimator=XGBRegressor())
data4 = imputer.fit_transform(data)
print(data4)
'''
[[ 2.          2.          2.          4.00096321]
 [ 2.00112057  4.          4.          4.        ]
 [ 6.          4.00000906  6.          4.00096321]
 [ 8.          8.          8.          8.        ]
 [10.          7.99906492 10.          7.99903679]]
'''
