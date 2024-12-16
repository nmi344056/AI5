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

from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

print(np.__version__)         # 1.22.4
# np.float = float            # 2.1.1 에서 사용시 추가

# pip install impyute
from impyute.imputation.cs import mice  # interpolate의 상향 버전
data9 = mice(data.values,
             n=10,
             seed=777)
print(data9)
'''
[[ 2.          2.          2.          1.99999928]
 [ 4.00000144  4.          4.          4.        ]
 [ 6.          6.01029603  6.          6.00343177]
 [ 8.          8.          8.          8.        ]
 [10.         10.04118411 10.         10.01372828]]
'''
