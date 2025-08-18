# -*- coding: utf-8 -*-
"""
Created on Mon Aug 18 12:17:56 2025

@author: human
"""

import pandas as pd
import numpy as np

data = pd.DataFrame(
    np.arange(6).reshape((2,3)),
    index = pd.Index(["Ohio","Colorado"], name = "state"),
    columns = pd.Index(["One","Two","Three"])
    )
data
result = data.stack()
result
result.shape
result.unstack()
result.unstack(level=0)
result.unstack(level=1)
result.unstack(level="state")

s1 = pd.Series([0,1,2,3],index=list("abcd"))
s2 = pd.Series([4,5,6],index=list("cde"))

data2 = pd.concat([s1,s2],keys=["one","two"])
data2.shape
data2.unstack()

# p 371
data = pd.DataFrame(
    np.arange(6).reshape((2,3)),
    index= pd.Index(["Ohio","Colorado"], name= "state"),
    columns = pd.Index(["One","Two","Three"])
    )
data
result = data.stack()
result
df = pd.DataFrame({
    "left":result,"right":result+5},
    columns=pd.Index(["left","right"],name="side"))
df
df.unstack(level="state")
df.unstack(level="state").stack(level="side")

### group by
df = pd.DataFrame(
    {
     "학생":["철수","영희","길동"],
     "수학":[90,85,70],
     "영어":[80,95,88],
     "과학":[85,77,92]
     }
    )
df
df[["수학","영어","과학"]].mean(axis=1)
pd.concat([
    df[["학생"]],
    df[["수학","영어","과학"]].mean(axis=1)],axis=1
    )

new_df=pd.melt(df, id_vars=["학생"]) #,var_name="과목"
new_df
new_df.groupby("학생")["value"].mean()
