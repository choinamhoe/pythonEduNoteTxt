
import glob, joblib, os
import pandas as pd 
import numpy as np 

# 원본데이터를 바탕으로 train.csv 파일 만들기

os.chdir("E:/cjcho_work/250923/235801_2021 농산물 가격예측 AI 경진대회")

files = glob.glob("public_data/train_AT_TSALET_ALL/*")
files = sorted(files)

file = files[0]
df = pd.read_csv(file)

# 도매 거래내역 모두 다를 포함
df.iloc[:,4].value_counts() 
# 1002 가지 정도의 다양한 품종 존재

# 대상 품목(16종)
unique_pum = [
    '배추', '무', '양파', '건고추','마늘',
    '대파', '얼갈이배추', '양배추', '깻잎',
    '시금치', '미나리', '당근', '파프리카', 
    '새송이', '팽이버섯', '토마토',
]
# 대상 품종(5종)
unique_kind = [
    '청상추', '백다다기', '애호박', 
    '캠벨얼리', '샤인마스캇'
]
df.columns

df["PUM_NM"] # 품목
df["KIND_NM"] # 품종
# 전체 데이터에서 대상 품목과 대상품종 추출
df2 = df[
    (df["PUM_NM"].isin(unique_pum))|(
        df["KIND_NM"].isin(unique_kind))]
df.shape # 270만건에서 56만건으로 추출
df2.describe()
# 거래량과 거래 금액에 음수가 있음
df2 = df2[df2["TOT_QTY"]>0]

np.where([True, False, False], 1, 2)
# 대상 품목과 대상 품종이 같이 있는 하나의 컬럼 생성(ITEM)
df2["ITEM"] = np.where(
    df2["PUM_NM"].isin(unique_pum),
    df2["PUM_NM"],df2["KIND_NM"])

## 30분 시작
# 판매일 및 대상별 평균 단가 거래량
agg_df = df2.groupby(
    ["SALEDATE", "ITEM"]).agg(
        TOT_QTY = ("TOT_QTY", "sum"),
        TOT_AMT = ("TOT_AMT", "sum")
        )
agg_df = agg_df.reset_index()
agg_df["PRICE"] = agg_df["TOT_AMT"] / agg_df["TOT_QTY"]

pivot_qty = agg_df.pivot(
    index = "SALEDATE", 
    columns = "ITEM", 
    values="TOT_QTY")
pivot_qty.columns= [f"{i}_거래량(kg)" for i in pivot_qty.columns]

pivot_price = agg_df.pivot(
    index = "SALEDATE", 
    columns = "ITEM", 
    values="PRICE")
pivot_price.columns= [f"{i}_가격(원/kg)" for i in pivot_price.columns]

pivot_qty= pivot_qty.reset_index()
pivot_price = pivot_price.reset_index()

final_df = pd.merge(pivot_qty,pivot_price,on="SALEDATE")
pd.to_datetime(final_df["SALEDATE"])
final_df["SALEDATE"] = pd.to_datetime(final_df["SALEDATE"].astype(str))
final_df["요일"] = final_df["SALEDATE"].dt.day_name(locale="ko_KR")

# 판매량이 없는 날짜 생성
temp_df = pd.DataFrame(
    {"SALEDATE":pd.date_range("2016-01","2016-01-31",freq="D")})
# 결측치 0으로 대체 & 요일 생성
final_df = pd.merge(final_df, temp_df, how="outer").fillna(0)
final_df["요일"] = final_df["SALEDATE"].dt.day_name(locale="ko_KR")
## 3시 20분 시작

final_df["SALEDATE"].dt.isocalendar().week.astype(int)
final_df["week"]=final_df["SALEDATE"].dt.isocalendar().week.astype(int)
final_df["kimjang"] = 0
final_df.loc[(46<=final_df["week"])&(final_df["week"]<=50),"kimjang"] =1

import holidays 
kr_holidays = holidays.KR(years=range(2016,2021))
final_df["holiday"] = final_df["SALEDATE"].isin(kr_holidays)+0

# 거래취소 건수? 취소된 거래량? 일별 취소된 거래량(거래 건수가 음수)
# 거래량이 음수인 대상 품목 및 품종
df3 = df[df["TOT_QTY"]<0]
df3 = df3[
    (df3["PUM_NM"].isin(unique_pum))|(
        df3["KIND_NM"].isin(unique_kind))]
# 품목 품종 하나의 컬럼으로 병합
df3["ITEM"] = np.where(
    df3["PUM_NM"].isin(unique_pum),
    df3["PUM_NM"],df3["KIND_NM"])

agg_df = df3.groupby(
    ["SALEDATE"]).agg(
        TOT_QTY = ("TOT_QTY", "sum"),
        )
agg_df = agg_df.reset_index()
