# -*- coding: utf-8 -*-
"""
Created on Wed Aug 20 09:41:20 2025

@author: human
"""

import glob, tqdm
import pandas as pd 

file_dir = "C:/Users/human/Downloads/조위_가덕도/조위_가덕도"
files = glob.glob(f"{file_dir}/*txt")
import random
len(files)
random.shuffle(files)
files = sorted(files)
file = files[0]

df = pd.read_csv(file, encoding="cp949", skiprows = 3, sep = '\t')
df["관측시간"] = pd.to_datetime(df["관측시간"])

# 5분까지 1월부터 12월까지 파일들 읽어와서 하나로 합치기 :)
dfs = list()
# for file in files:
for file in tqdm.tqdm(files):
    df = pd.read_csv(file, encoding="cp949", skiprows = 3, sep = '\t')
    df["관측시간"] = pd.to_datetime(df["관측시간"])
    dfs.append(df)
tot_df = pd.concat(dfs,ignore_index=True)
#df["관측시간"].dt.minute
### 20분까지 분단위 데이터를 시간단위 데이터로 추출하기
# type 1 
tot_df.loc[tot_df["관측시간"].dt.minute==0,:]
# type 2
values = tot_df["관측시간"].astype(str)
values = [i[14:16]=="00" for i in values]
tot_df.loc[values,:]
# type 3
tot_df.loc[:, "구분자"] = tot_df["관측시간"].astype(str)
tot_df.loc[:, "구분자"] = [i[:13] for i in tot_df.loc[:,"구분자"]]
tot_df.drop_duplicates('구분자', keep="first")

# type 1 
dfs = list()
for file in tqdm.tqdm(files):
    df = pd.read_csv(file, encoding="cp949", skiprows = 3, sep = '\t')
    df["관측시간"] = pd.to_datetime(df["관측시간"])
    df = df.loc[df["관측시간"].dt.minute==0,:]
    dfs.append(df)
tot_df = pd.concat(dfs,ignore_index=True)
# type 2 
dfs = list()
for file in tqdm.tqdm(files):
    df = pd.read_csv(file, encoding="cp949", skiprows = 3, sep = '\t')
    values = [i[14:16]=="00" for i in df["관측시간"]]
    df = df.loc[values,:]
    df["관측시간"] = pd.to_datetime(df["관측시간"])
    dfs.append(df)
tot_df = pd.concat(dfs,ignore_index=True)


#### 44분까지 matplotlib 활용해서 수온 시각화 plot 함수 활용
import matplotlib.pyplot as plt 
%matplotlib auto
fig, ax1 = plt.subplots(figsize=(10,5))
tot_df.columns
ax1.plot(tot_df["관측시간"], tot_df["수온(℃)"])
# 50분까지 - 결측값을 np.nan 으로 변경 수온 타입을 np.float32로 변경
import numpy as np
logic = tot_df.loc[:,"수온(℃)"]=="-"
tot_df.loc[logic,"수온(℃)"]=np.nan
tot_df.loc[:,"수온(℃)"]
tot_df.loc[:,"수온(℃)"] = tot_df.loc[:,"수온(℃)"].astype(np.float32)

ax1.cla()
ax1.plot(tot_df["관측시간"], tot_df["수온(℃)"])

logic1 = tot_df.loc[:,"관측시간"]< pd.Timestamp("2024-04-01")
logic2 = tot_df.loc[:,"수온(℃)"]>14
tot_df.loc[logic1&logic2,"수온(℃)"] = np.nan

ax1.cla()
ax1.plot(tot_df["관측시간"], tot_df["수온(℃)"])

logic1 = tot_df.loc[:,"관측시간"]>pd.Timestamp("2024-09-01")
logic2 = tot_df.loc[:,"관측시간"]<pd.Timestamp("2024-10-11")
logic3 = tot_df.loc[:,"수온(℃)"]<23
tot_df.loc[logic1&logic2&logic3,["관측시간","수온(℃)"]]
tot_df.loc[logic1&logic2&logic3,"수온(℃)"] = np.nan
ax1.cla()
ax1.plot(tot_df["관측시간"], tot_df["수온(℃)"])

# 30 분까지 기온 우측에 그려보기 :)
ax2 = ax1.twinx()
(tot_df["기온(℃)"]=="-").sum()
tot_df.loc[tot_df["기온(℃)"]=="-","기온(℃)"]= np.nan
tot_df["기온(℃)"] = tot_df["기온(℃)"].astype(np.float32)
ax2.cla()
ax2.plot(
    tot_df["관측시간"], tot_df["기온(℃)"], color = "red", alpha=0.5)

ax3 = ax1.twinx()
tot_df.loc[tot_df["풍속(m/s)"]=="-","풍속(m/s)"] = np.nan
(tot_df["풍속(m/s)"]=="-").sum()
tot_df["풍속(m/s)"] = tot_df["풍속(m/s)"].astype(np.float32)

ax3.plot(
    tot_df["관측시간"], tot_df["풍속(m/s)"], 
    color = "green", alpha=0.5)


########################
#api 로 데이터 가져오기
########################
import json, requests
key = "BehDhlIyPwKozERUb2BYQ=="
BASE_URL = "http://www.khoa.go.kr/api"
SUB_URL = "oceangrid/ObsServiceObj/search.do"


URL = f"{BASE_URL}/{SUB_URL}?ServiceKey={key}&ResultType=json"
res = requests.get(URL)
res.status_code
#res.text
res_data = res.json()
res_data.keys()

### 메타데이터
res_data["result"].keys()
df = pd.DataFrame(res_data["result"]["data"])
df

### 조위 관측 데이터 
SUB_URL = "oceangrid/tideObs/search.do"
station_id = "DT_0063"
date = "20241011"
URL = f"{BASE_URL}/{SUB_URL}?ServiceKey={key}&ObsCode={station_id}&Date={date}&ResultType=json"
res = requests.get(URL)
res_data = res.json()
obs_df = pd.DataFrame(res_data["result"]["data"])

## 35분까지 geopandas  패키지랑 contextily 패키지 활용해서 지적도 시각화
## 여유되면 관측소 위치까지 시각화
import geopandas as gpd
import contextily as ctx
shp_file = "C:/Users/human/Downloads/ctprvn_20230729/ctprvn.shp"
gdf = gpd.read_file(shp_file)
gdf = gdf.set_crs("EPSG:5179", allow_override=True)
gdf.to_crs(epsg=4326)
gdf = gdf.to_crs(epsg=3857)

ax = gdf.plot()
#OpenWeatherMap
ctx.add_basemap(ax, source=ctx.providers.OpenRailwayMap)
# source = ctx.providers.OpenStreetMap.Mapnik

location_gdf = gpd.GeoDataFrame(
    df.drop(["obs_lat","obs_lon"], axis= 1), 
            geometry= gpd.points_from_xy(
                df.loc[:,"obs_lon"], df.loc[:,"obs_lat"]))
location_gdf = location_gdf.set_crs("EPSG:4326", allow_override=True)
location_gdf = location_gdf.to_crs(epsg=3857)

location_gdf.plot(ax= ax, color= "red", markersize= 1)

"""
참고
https://m.blog.naver.com/dsz08082/222801593268
"""
import folium
m = folium.Map(
    location=[37.5665, 126.9780], 
    zoom_start=12, tiles="cartodb positron") 
m.save('map.html')

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

folium.Circle(
    [37.50, 127.003], radius=100,color='#ABF200',
    fill=True, fill_color='#ABF200').add_to(m)
folium.Circle(
    [37.50, 126.997], radius=200,color='#FF0000',
    fill=True, fill_color='#FF0000').add_to(m)
m.save('map.html')

import geopandas as gpd
import contextily as ctx
shp_file = "C:/Users/human/Downloads/ctprvn_20230729/ctprvn.shp"
gdf = gpd.read_file(shp_file, encoding="cp949")
gdf = gdf.set_crs("EPSG:5179", allow_override=True)
gdf = gdf.to_crs(epsg=4326)

fields = [i for i in gdf.columns if i != "geometry"]

folium.GeoJson(
    gdf, name="shp", 
    tooltip=folium.GeoJsonTooltip(
        fields=fields)
    ).add_to(m)
m.save('map.html')
import sklearn, seaborn, folium
