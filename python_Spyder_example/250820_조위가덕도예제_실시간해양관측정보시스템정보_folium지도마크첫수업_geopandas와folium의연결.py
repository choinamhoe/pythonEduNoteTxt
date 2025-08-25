# -*- coding: utf-8 -*-
"""
Created on Wed Aug 20 09:18:16 2025

@author: human
"""

import glob,tqdm
import pandas as pd

selected_columns = ['관측시간', '조위(cm)', '수온(℃)', '염분(PSU)', '유의파고(m)', '유의파주기(sec)',
       '최대파고(m)', '최대파주기(sec)', '풍속(m/s)', 'Unnamed: 9', '풍향(deg)', '기온(℃)',
       '기압(hPa)', '시정(m)']


file_dir = "E:/최남회/250813_조창제강사_해양자료시각화_심평원인공지능자료/조위_가덕도"
files = glob.glob(f"{file_dir}/*.txt")
files
file = files[0]
file

df = pd.read_csv(file,encoding="cp949")
df
df = pd.read_csv(file,encoding="cp949",skiprows = 3, sep = "\t")
df.columns
df
df["관측시간"] = pd.to_datetime(df["관측시간"])

#1월부터 12월까지 파일을 읽어와서 하나로 합치기
#가덕도에 있는 모든.txt파일을 읽어서 합치는 작업을 dt_df에 할당
dfs = list()
for file in tqdm.tqdm(files):
    df = pd.read_csv(file,encoding="cp949",skiprows = 3, sep = "\t")
    df["관측시간"] = pd.to_datetime(df["관측시간"])
    dfs.append(df)
tot_df = pd.concat(dfs,axis=0,ignore_index=True)
tot_df
print(dir(tot_df["관측시간"].dt))
tot_df["관측시간"].dt.minute
tot_df["관측시간"].dt.hour
#분단위 데이터를 시간단위 데이터로 추출하기
#type1
tot_df.loc[tot_df["관측시간"].dt.minute==0,:]
#type2
values = tot_df["관측시간"].astype(str)
values = [i[14:16] == "00" for i in values]
tot_df.loc[values,:]

#type3
tot_df.loc[:,"구분자"] = tot_df["관측시간"].astype(str)
tot_df.loc[:,"구분자"] = [i[13] for i in tot_df.loc[:,"구분자"]]
tot_df.drop_duplicates('구분자',keep="first")
tot_df

#애초부터 관측시간에 0분으로 다시 매핑 작업
#type1
dfs = list()
for file in tqdm.tqdm(files):
    df = pd.read_csv(file,encoding="cp949",skiprows = 3, sep = "\t")
    df["관측시간"] = pd.to_datetime(df["관측시간"])
    df = df.loc[df["관측시간"].dt.minute==0,:]
    dfs.append(df)
tot_df = pd.concat(dfs,axis=0,ignore_index=True)
tot_df
#type2
dfs = list()
for file in tqdm.tqdm(files):
    df = pd.read_csv(file,encoding="cp949",skiprows = 3, sep = "\t")
    values = [i[14:16] == "00" for i in df["관측시간"]]
    df = df.loc[values,:]
    df["관측시간"] = pd.to_datetime(df["관측시간"])
    dfs.append(df)
tot_df = pd.concat(dfs,axis=0,ignore_index=True)
tot_df
tot_df.columns   

#### 44분까지 matplotlib 활용해서 수온 시각화 plot 함수 활용
import matplotlib.pyplot as plt
%matplotlib auto
fig, ax1 = plt.subplots(figsize = (10,5))
ax1.plot(tot_df["관측시간"],tot_df["수온(℃)"],label="Temperature(°C)",color="red")
ax1.set_ylabel("Temperature(°C)",color="red",fontsize="15")
ax1.legend(loc="upper left",fontsize="10") # (upper, center, lower),(left, center,right)
# 50분까지 - 결측값을 np.nan 으로 변경 수온 타입을 np.float32로 변경
import numpy as np
logic = tot_df.loc[:,"수온(℃)"]=="-"
tot_df.loc[logic,"수온(℃)"]=np.nan
tot_df.loc[:,"수온(℃)"]
tot_df.loc[:,"수온(℃)"] = tot_df.loc[:,"수온(℃)"].astype(np.float32)

ax1.cla()
ax1.plot(tot_df["관측시간"],tot_df["수온(℃)"],label="Water Temperature(°C)",color="red")
ax1.set_ylabel("Water Temperature(°C)",color="red",fontsize="15")
ax1.legend(loc="upper left",fontsize="10") # (upper, center, lower),(left, center,right)


logic1 = tot_df.loc[:,"관측시간"]< pd.Timestamp("2024-04-01")
logic2 = tot_df.loc[:,"수온(℃)"]>14
tot_df.loc[logic1&logic2,"수온(℃)"] = np.nan

ax1.cla()
ax1.plot(tot_df["관측시간"],tot_df["수온(℃)"],label="Water Temperature(°C)",color="blue")
ax1.set_ylabel("Water Temperature(°C)",color="blue",fontsize="15")
ax1.legend(loc="upper left",fontsize="10") # (upper, center, lower),(left, center,right)


logic1 = tot_df.loc[:,"관측시간"]>pd.Timestamp("2024-09-01")
logic2 = tot_df.loc[:,"관측시간"]<pd.Timestamp("2024-10-11")
logic3 = tot_df.loc[:,"수온(℃)"]<23
tot_df.loc[logic1&logic2&logic3,["관측시간","수온(℃)"]]
tot_df.loc[logic1&logic2&logic3,"수온(℃)"] = np.nan
ax1.cla()
ax1.plot(tot_df["관측시간"],tot_df["수온(℃)"],label="Water Temperature(°C)",color="blue")
ax1.set_ylabel("Water Temperature(°C)",color="blue",fontsize="15")
ax1.legend(loc="upper left",fontsize="10") # (upper, center, lower),(left, center,right)

#30분까지 기온 우측에 그려보기
ax2 = ax1.twinx()
tot_df.columns
(tot_df["기온(℃)"]=="-").sum()
tot_df.loc[tot_df["기온(℃)"]=="-","기온(℃)"] = np.nan
tot_df["기온(℃)"] = tot_df["기온(℃)"].astype(np.float32)
ax2.cla()
ax2.plot(tot_df["관측시간"],tot_df["기온(℃)"],label="Temperature(°C)",color="red")
ax2.set_ylabel("Temperature(°C)",color="red",fontsize="15")
ax2.legend(loc="upper right",fontsize="10") # (upper, center, lower),(left, center,right)
#2번째 y축 레이블이 오른쪽으로 오게 설정
ax2.yaxis.set_label_position("right")

ax3 = ax1.twinx()
(tot_df["풍속(m/s)"]=="-").sum()
tot_df.loc[tot_df["풍속(m/s)"]=="-","풍속(m/s)"] = np.nan
tot_df["풍속(m/s)"] = tot_df["풍속(m/s)"].astype(np.float32)
ax3.cla()
ax3.plot(tot_df["관측시간"],tot_df["풍속(m/s)"],label="wind Speed(m/s)",color="green")
#ax3.set_ylabel("wind Speed(m/s)",color="green",fontsize="15")
ax3.legend(loc="upper center",fontsize="10") # (upper, center, lower),(left, center,right)
#2번째 y축 레이블이 오른쪽으로 오게 설정
ax3.yaxis.set_label_position("right")


#####################################
#api로 데이터 가져오기
#실시간해양관측정보 시스템 : https://www.khoa.go.kr/oceangrid/koofs/kor/observation/obs_real.do
#회원가입 후 마이페이지에서 발급받은키 확인 가능
#openapi>>관측소>>관측소별 데이터제공현황 주소 : 아래 주소
#http://www.khoa.go.kr/api/oceangrid/ObsServiceObj/search.do?ServiceKey=o8mt3lbEqOxec6R2V/BfHw==&ResultType=json

#openapi>>조위>>조위관측소 실측 조위 : 아래 주소
#http://www.khoa.go.kr/api/oceangrid/tideObs/search.do?ServiceKey=wldhxng34hkddbsgm81lwldhxng34hkddbsgm81l==&ObsCode=DT_0001&Date=20230101&ResultType=json
#발급받은 키 : o8mt3lbEqOxec6R2V/BfHw==
#####################################
import pandas as pd
import json, requests
key = "o8mt3lbEqOxec6R2V/BfHw=="
BASE_URL = "http://www.khoa.go.kr/api"
SUB_URL = "oceangrid/ObsServiceObj/search.do"

URL = f"{BASE_URL}/{SUB_URL}?ServiceKey={key}&ResultType=json"
res = requests.get(URL)
res.status_code
res.text
res_data = res.json()
res_data
res_data.keys()
###메타데이터
res_data['result'].keys()
df = pd.DataFrame(res_data['result']['data'])
df
"""
Index(['obs_post_id', 'obs_lat', 'data_type', 'obs_post_name', 'obs_lon',
       'obs_object'],
      dtype='object')
"""
#조위 관측 데이터(가덕도 - DT_0063)
SUB_URL = "oceangrid/tideObs/search.do"
station_id = "DT_0063"
date = "20241011"
URL = f"{BASE_URL}/{SUB_URL}?ServiceKey={key}&ObsCode={station_id}&Date={date}&ResultType=json"
res = requests.get(URL)
res_data = res.json()
obs_df = pd.DataFrame(res_data['result']['data'])
obs_df
obs_df.columns

## 35분까지 geopandas  패키지랑 contextily 패키지 활용해서 지적도 시각화
## 여유되면 관측소 위치까지 시각화
import pandas as pd
import geopandas as gpd
import contextily as ctx
shp_file = "E:/최남회/python_Spyder_example/20250819/ctprvn_20230729/ctprvn.shp"
gdf = gpd.read_file(shp_file)
gdf = gdf.set_crs("EPSG:5179", allow_override=True)
gdf.to_crs(epsg=4326)
gdf = gdf.to_crs(epsg=3857)

ax = gdf.plot()
#OpenWeatherMap
ctx.add_basemap(ax, source=ctx.providers.OpenRailwayMap)
# source = ctx.providers.OpenStreetMap.Mapnik
dir(ctx.providers)
dir(ctx.providers.OpenRailwayMap)
x = df.loc[:,'obs_lon']
y = df.loc[:,'obs_lat']
location_df = gpd.GeoDataFrame(
    df.drop(["obs_lat","obs_lon"],axis= 1), geometry=gpd.points_from_xy(
        x
        , y
        ))
location_df = location_df.set_crs("EPSG:4326", allow_override=True)
location_df = location_df.to_crs(epsg=3857)
location_df.plot(ax = ax, color = "green")
"""
참고
https://m.blog.naver.com/dsz08082/222801593268
"""
import folium

import sklearn, seaborn, folium

#m = folium.Map(location=[37.5665, 126.9780], zoom_start=12)
m = folium.Map(location=[37.5665, 126.9780],
               zoom_start=12, 
               tiles="cartodb positron") 
#m.save('map.html')

folium.Marker(
    location=[37.5665, 126.9780],
    popup='서울 시청',
    tooltip='Click me'
).add_to(m)
m.save('map.html')

tooltip = "클릭해주세요 :)"
folium.Marker(
    [37.503, 127.001],
    popup='<i>위</i>', 
    tooltip=tooltip).add_to(m)
folium.Marker(
    [37.50, 127.001], 
    popup='가운데', 
    tooltip=tooltip).add_to(m)
folium.Marker(
    [37.497, 127.001], 
    popup='<b>아래</b>',
    tooltip=tooltip).add_to(m) #b는 굵게
m.save('map.html')

folium.Marker(
    [37.503, 127.001],
    icon=folium.Icon(icon='cloud')).add_to(m)
folium.Marker(
    [37.50, 127.001],
    icon=folium.Icon(
        icon='home',color='green')).add_to(m)
folium.Marker(
    [37.497, 127.001],
    icon=folium.Icon(
        icon='info-sign',color='red')).add_to(m) 
m.save('map.html')

folium.Marker([37.503, 127.001], icon=folium.Icon(icon='cloud')).add_to(m)
folium.Marker([37.50, 127.001],icon=folium.Icon(icon='home',color='green')).add_to(m)
folium.Marker([37.497, 127.001], icon=folium.Icon(icon='info-sign',color='red')).add_to(m) 
folium.Circle([37.50, 127.003], radius=100,color='#ABF200',fill=True, fill_color='#ABF200').add_to(m)
folium.Circle([37.50, 126.997], radius=200,color='#FF0000',fill=True, fill_color='#FF0000').add_to(m) 
m.save('map.html')

import geopandas as gpd
import contextily as ctx
shp_file = "E:/최남회/python_Spyder_example/20250819/ctprvn_20230729/ctprvn.shp"
gdf = gpd.read_file(shp_file, encoding="cp949")
gdf = gdf.set_crs("EPSG:5179", allow_override=True)
gdf = gdf.to_crs(epsg=4326)
m = folium.Map(location=[37.5665, 126.9780],
               zoom_start=12, 
               tiles="cartodb positron") 
fields = [c for c in gdf.columns if c != "geometry"]
folium.GeoJson(gdf, name="shp",
               tooltip=folium.GeoJsonTooltip(fields=fields)
               ).add_to(m)
folium.Marker(
    location=[37.5665, 126.9780],
    popup='서울 시청',
    tooltip='Click me'
).add_to(m)
m.save('map.html')
