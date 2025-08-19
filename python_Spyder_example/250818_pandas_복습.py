# -*- coding: utf-8 -*-
"""
Created on Mon Aug 18 10:32:57 2025

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

df = pd.DataFrame(
    {"col1":[4000,5500,7000],
     "col2":["가","나","다"]},index=["a","b","c"])
df
df.iloc[1,1]
df.loc["b","col2"]

df = pd.DataFrame(
    {"col1":[4000,5500,7000],
     "col2":["가","나","다"]})
df
df.loc[1,"col2"]
df.loc[1:]
df2 = df.loc[1:]
df2
df2.loc[1,:]
df2.loc[1]
df2 = df2.reset_index(drop=True)
df2
df2.loc[1,:]
df2.loc[1]

"""
df.drop: 행이나 열 삭제 할 때 사용
df.reset_index: 인덱스 값을 초기화 할 때 사용
"""
#axis 0, 1, rows, columns
df.drop(1,axis=0) 
df.drop("col2",axis=1) 
df.reset_index()


"""

df.to_numpy, df.to_dict
df.isna : 결측 존재 유무 확인
df.isin : 특정값 포함 유무 확인
df.value_count : 특정 값 갯수 확인
"""
import numpy as np
df = pd.DataFrame(
    {"col1":[4000,5500,np.nan, 5000],
     "col2":["가","나","다","가"]},index=["a","b","c","d"])
df
df.isna()
df["col1"].isna()
df["col2"].isin(["가","다"])
df.loc[df["col2"].isin(["가","다"]),:]
df.loc[df["col2"].isin(["가","다"])]
df["col2"].value_counts()
df
"""
df.dropna : 결측치를 포함하는 행을 제거
df.fillna : 결측치를 특정값으로 대체
df.drop_duplicates : 중복되는 행을 제거
"""
df
df.dropna()
df.fillna(-999)
df.drop_duplicates()
df.drop_duplicates("col2")
df.drop_duplicates("col2", keep="last")
df.drop_duplicates("col2", keep="first")

"""
concat, merge
"""
df1 = pd.DataFrame(
    np.random.randn(12).reshape((3,4)),
    columns = list("abcd")
    )
df2 = pd.DataFrame(
    np.random.randn(6).reshape((2,3)),
    columns = list("bda")
    )
df1
df2
pd.concat([df1,df2])
# 인덱스를 0부터 생성하고 싶을 때
pd.concat([df1,df2], ignore_index=True) 
pd.concat([df1,df2],axis=1)

### merge
df1 = pd.DataFrame(
    {
     "key":list("bbacaab"),
     "data1":pd.Series(range(7))
     })
df2 = pd.DataFrame(
    {
     "key":list("abd"),
     "data2":pd.Series(range(3))
     })
df1
df2
pd.merge(df1,df2,on="key",how="left")
pd.merge(df1,df2,on="key",how="right")
pd.merge(df1,df2,on="key",how="outer")
pd.merge(df1,df2,on="key",how="inner")

df1.columns = ["l_key","data1"]
df2.columns = ["r_key","data2"]
pd.merge(df1,df2,left_on="l_key",right_on="r_key",how="inner")
"""
오늘 배울 내용
df.get_dummies : 카테고리를 컬럼으로 만들고 True/False형태로 반환
df.melt : 컬럼을 행으로 변환
df.pivot : 행을 컬럼형태로 변환
"""
df = pd.DataFrame(
    {
     "학생":["철수","영희","길동"],
     "수학":[90,85,70],
     "영어":[80,95,88],
     "과학":[85,77,92]
     }
    )
df
new_df=pd.melt(df, id_vars=["학생"]) #,var_name="과목"
new_df
pd.get_dummies(new_df, columns=["variable"],dtype=np.int32)
pd.get_dummies(new_df, columns=["variable"],dtype=np.int32)
new_df.pivot(index="학생",columns="variable")
"""
----------------------------------------------------------------------
pd.Series : 단일 컬럼
pd.DataFrame : 다중 컬럼
df.head,df.tail

indexing
iloc(integer location)
loc(Label based Location)
df.drop
df.reset_index

df.to_numpy, df.to_dict
df.isna, df.isin,df.value_count
df.dropna, df.fillna, df.drop_duplicates

오늘 배울 내용
df.get_dummies
df.melt
df.pivot
"""