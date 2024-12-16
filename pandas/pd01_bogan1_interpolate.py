'''
결측치 처리
1. 삭제 - 행 또는 열
2. 임의의 값
    평균 : mean (평균의 오류 : 이상치)
    중위 : median
    0 : fillna
    앞값 : ffill
    뒷값 : bfill
    특정값 : 777 (조건을 같이 넣는다.)
    기타등등
3. bogan (보간법)
4. 모델 : .predict (같은 모델을 사용하면 과적합, 다른 모델 사용)
5. 부스팅 계열 모델 : 통상 결측치, 이상치에 대해 자유롭다.
'''

import pandas as pd
import numpy as np

dates = ['10/11/2024', '10/12/2024', '10/13/2024', 
         '10/14/2024', '10/15/2024', '10/16/2024', ]
dates = pd.to_datetime(dates)
print(dates)
# DatetimeIndex(['2024-10-11', '2024-10-12', '2024-10-13', '2024-10-14',
#                '2024-10-15', '2024-10-16'],
#               dtype='datetime64[ns]', freq=None)

ts = pd.Series([2, np.nan, np.nan, 8, 10, np.nan], index = dates)
print(ts)
'''
2024-10-11     2.0
2024-10-12     NaN
2024-10-13     NaN
2024-10-14     8.0
2024-10-15    10.0
2024-10-16     NaN
dtype: float64
'''

ts = ts.interpolate()
print(ts)
'''
2024-10-11     2.0
2024-10-12     4.0
2024-10-13     6.0
2024-10-14     8.0
2024-10-15    10.0
2024-10-16    10.0    뒷 값은 마지막 값으로 ffill -> 조심해야 한다
dtype: float64
'''
