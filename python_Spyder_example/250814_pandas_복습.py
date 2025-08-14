# -*- coding: utf-8 -*-
"""
Created on Thu Aug 14 11:23:18 2025

@author: human
"""
import pandas as pd

"""
pd.Series : 단일 컬럼
pd.DataFrame : 다중 컬럼
"""
df = pd.Series([4000,5500,7000],index=["a","b","c"])
df
df = pd.DataFrame(
    {
     "col1":[4000,5500,7000],
     "col2":["가","나","다"]
     },index=["a","b","c"]
    )
df
my_list = [4000,5500,7000]
pd.Series(my_list)
my_dict = {
         "col1":[4000,5500,7000],
         "col2":["가","나","다"]
         }
pd.DataFrame(my_dict)
    
"""
pd.Series : 단일 컬럼
pd.DataFrame : 다중 컬럼
# index, label
iloc(integer location)
loc(label based location)
df.iloc[3:8,1:3]
df.loc[:,["컬럼명1","컬럼명2"]]
df.index,df.columns
"""
df = pd.Series([4000,5500,7000],index=["a","b","c"])
df
df.iloc[:2]
df.loc[["a","c"]]
df = pd.DataFrame(
    {
     "col1":[4000,5500,7000],
     "col2":["가","나","다"]
     },index=["a","b","c"]
    )
df
df.iloc[:2,:1]
df.loc[["a","c"],"col1"]
"""
df.index: 행에 대한 이름
df.column: 열에 대한 이름
df.values : 값
"""
df.index
list(df.index)
df.columns
list(df.columns)
df.values
list(df.values)

"""
df.drop: 특정 행이나 열을 삭제 하고 싶은 경우 사용
df.reset_index : index 값을 초기화
(기존 키 값은 새로운 컬럼으로 생성)
df.reset_index(drop=True) : 기존 키 값은 삭제
"""
df = pd.DataFrame(
    {
     "col1":[4000,5500,7000],
     "col2":["가","나","다"]
     },index=["a","b","c"]
    )
df
df.reset_index()
df.reset_index(drop=True)
#axis=0, 1, rows, columns 사용 가능
df.drop(["col1"],axis="columns")
df.drop(["a","c"], axis=0)
"""
df.to_numpy : 값 추출,
df.to_dict : 딕셔너리 형태로 추출
"""
df.values
#df.to_numpy()
df.to_dict()
"""
df.isna : 결측인지 유무 확인
df.isin : 포함되어 있는지 유무 확인
"""
import numpy as np
df = pd.DataFrame(
    {
     "col1":[4000,5500,np.nan],
     "col2":["가","나","다"]
     },index=["a","b","c"]
    )
df.isna()
df["col1"].isna() #df.loc[:,"col1"].isna()
df["col2"].isin(["가","다"])


# col1 컬럼이 결측인 행 추출
df.loc[df["col1"].isna(),:]
df[df["col1"].isna()]
# col2 컬럼이 가 다가 포함된 행 추출
df.loc[df["col2"].isin(["가","다"]),:]
df[df["col2"].isin(["가","다"])]
"""
df.mean: 평균 계산
df.std: 표준편차 계산
"""
df = pd.DataFrame(
    {"col1":[4000,5500,np.nan],
     "col2":["가","나","다"],
     "col3":[3000,5000,1600],
     "col4":[3000,5000,1600],
     },index=["a","b","c"])

df.loc[:,["col1","col3"]].mean()
df.loc[:,["col1","col3"]].mean(axis=0) # 열 평균
df.loc[:,["col1","col3"]].mean(axis=1) # 행 평균
df.loc[:,["col1","col3"]].std()
df.loc[:,["col1","col3"]].std(axis=0)
df.loc[:,["col1","col3"]].std(axis=1)
df.loc[:,["col1","col3","col4"]].std(axis=1)
df.loc[:,["col1","col3","col4"]].std(
    axis=1,skipna=False)

"""
df.dropna : 결측 행 제거
df.fillna : 결측 값을 특정 값으로 대체
df.drop_duplicates : 중복되는 행 중 한개만 남김
## keep 옵션은 first last 등이 존재
"""
df = pd.DataFrame(
    {"col1":[4000,5500,np.nan],
     "col2":["가","나","가"],
     },index=["a","b","c"])
df
df.dropna()
df["col1"].dropna()
df.fillna(-999)
df.drop_duplicates("col2",keep="last")
df.drop_duplicates("col2",keep="first")

df.drop_duplicates()
df = pd.DataFrame(
    {"col1":[4000,5500,4000],
     "col2":["가","나","가"],
     },index=["a","b","c"])
df.drop_duplicates()


"""
df.head(n) : 위에서 n개 출력
df.tail(n) : 밑에서 n개 출력
df.T : 행과 열 변경
df.dtypes : 컬럼 타입 확인
"""
df = pd.DataFrame(
    {
     "col1":[4000,5500,4000],
     "col2":["가","나","가"]
     },index=["a","b","c"]
    )
df.head(2)
df.tail(1)
df.dtypes # int64, object
t_df = df.T
t_df.dtypes # object, object, object

"""
pd.Series : 단일 컬럼
pd.DataFrame : 다중 컬럼
# index, label
df.iloc[3:8,1:3]
df.loc[:,["컬럼명1","컬럼명2"]]
df.index,df.columns,df.values
df.drop,df.reset_index
df.to_numpy,df.to_dict
df.mean,df.std
"""
