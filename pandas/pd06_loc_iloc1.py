import pandas as pd
print(pd.__version__)   # 1.3.4

data = [
    ["삼성", "1000", "2000"],
    ["현대", "1100", "3000"],
    ["LG", "2000", "500"],
    ["아모레", "3500", "6000"],
    ["네이버", "100", "1500"],
]

index = ['031', '059', '033', '045', '023']
columns = ['종목명', '시가', '종가']

df = pd.DataFrame(data=data, index=index, columns=columns)

# print(df)
'''
     종목명    시가    종가
031   삼성  1000  2000
059   현대  1100  3000
033   LG  2000   500
045  아모레  3500  6000
023  네이버   100  1500
'''
print("==============================")
# print(df[0])    # raise KeyError(key) from err
# print(df['031'])    # raise KeyError(key) from err, '031' 이라는 컬럼을 출력하라는 명령어
print(df['시가'])       # 'Pandas 열행'이기 때문에 ***컬럼이 기준***

##### 아모레를 출력하려면 #####
# print(df[3, 0])     # KeyError: (3, 0)
# print(df['045', '종목명'])  # KeyError: ('045', '종목명')
# print(df['종목명', '045'])  # KeyError: ('종목명', '045')
print(df['종목명']['045'])  # 아모레
print("==============================")

##################################################
# loc : 인덱스를 기준으로 행 데이터 추출
# iloc : 행번호를 기준으로 행 데이터 추출 (int location으로 외우기)

##################################################

##### 아모레를 출력하려면 #####
# print(df.iloc['045'])   # TypeError: Cannot index by location index with a non-integer key
print(df.iloc[3])   # ㅇ

# print(df.loc[3])   # raise KeyError(key) from err
print(df.loc['045'])   # ㅇ


print("========== 네이버 출력 ==========")
print(df.loc['023'])
print(df.iloc[4])

print("========== 아모레 종가 출력 ==========")
print(df.loc['045']['종가'])        # 6000
print(df.loc['045', '종가'])        # 6000
print(df.loc['045'].loc['종가'])    # 6000
# print(df.iloc[4])

print(df.iloc[3][2])        # 6000 (pandas 2에서는 warring 있다.)
print(df.iloc[3, 2])        # 6000
print(df.iloc[3].iloc[2])   # 6000

print(df.loc['045'][2])     # 6000
# print(df.loc['045', 2])     # raise KeyError(key) from err

print(df.iloc[3]['종가'])     # 6000
# print(df.iloc[3, '종가'])     # ValueError: Location based indexing can only have [integer, integer slice (START point is INCLUDED, END point is EXCLUDED), listlike of integers, boolean array] types

print(df.loc['045'].iloc[2])    # 6000
print(df.iloc[3].loc['종가'])   # 6000

# 인덱스명, 컬럼명
# 인덱스 값이 아니라 순서다
