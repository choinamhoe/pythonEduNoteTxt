# -*- coding: utf-8 -*-
"""
Created on Tue Aug 19 15:29:58 2025

@author: human
"""
import matplotlib.pyplot as plt
%matplotlib auto
fig, ax = plt.subplots(figsize = (10,5))

### 여기는 지도가 다 표시가 안되서 주석처리
#import geopandas as gpd
#shp_file = "E:/최남회/python_Spyder_example/20250819/ctprvn_20230729/ctprvn.shp"
#gdf = gpd.read_file(shp_file)
#gdf = gdf.set_crs("EPSG:5179", allow_override=True)
#gdf = gdf.to_crs(epsg=4326)
#ax = gdf.plot()
#import contextily as ctx
#ctx.add_basemap(ax, source= ctx.providers.OpenStreetMap.Mapnik)

import geopandas as gpd
shp_file = "E:/최남회/python_Spyder_example/20250819/ctprvn_20230729/ctprvn.shp"
gdf = gpd.read_file(shp_file)
gdf = gdf.set_crs("EPSG:5179", allow_override=True)
gdf = gdf.to_crs(epsg=4326)
gdf = gdf.to_crs(epsg=3857)
ax = gdf.plot()
import contextily as ctx
ctx.add_basemap(ax, source= ctx.providers.OpenStreetMap.Mapnik)
import pandas as pd
df = pd.read_csv("E:/최남회/파이썬개발에대한파일모음/기상철 기상자료개발포털/20250819/지점정보_20250819.csv",encoding="CP949")
df.columns
"""
Index(['지점', '시작일', '종료일', '지점명', '지점주소', '관리관서', '위도', '경도', '노장해발고도(m)',
       '기압계(관측장비지상높이(m))', '기온계(관측장비지상높이(m))', '풍속계(관측장비지상높이(m))',
       '강우계(관측장비지상높이(m))'],
      dtype='object')
"""
df = df.loc[:,['지점','지점명','위도', '경도','노장해발고도(m)']]
df
x = df.loc[:,'경도']
y = df.loc[:,'위도']
location_df = gpd.GeoDataFrame(
    df, geometry=gpd.points_from_xy(
        x
        , y
        ))
location_df
location_df = location_df.set_crs("EPSG:5179", allow_override=True)
location_df = location_df.to_crs(epsg=4326)
location_df = location_df.to_crs(epsg=3857)
location_df.plot(ax = ax, color = "red")

