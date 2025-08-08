# -*- coding: utf-8 -*-
"""
Created on Fri Aug  8 14:10:15 2025

@author: human
"""

from sqlalchemy import create_engine, text
import pandas as pd
import requests
db_id = "root"
db_password = "test"
host = "localhost"
port = 3307
db_info = f"mysql+pymysql://{db_id}:{db_password}@{host}:{port}/auth"
engine = create_engine(
    db_info, connect_args={}
)
API_KEY = "K66mUw9zTByuplMPczwchQ"
BASE_URL = "https://apihub.kma.go.kr/api/typ01/url/"
SUB_URL = "kma_sfctm3.php?"
start_dt = "2015-12-11 01:00"
end_dt = "2015-12-12 23:00"
start_dt = pd.to_datetime(start_dt).strftime("%Y%m%d%H%M")
end_dt = pd.to_datetime(end_dt).strftime("%Y%m%d%H%M")
station_number = 108
f"{BASE_URL}{SUB_URL}tm1={start_dt}&tm2={end_dt}&stn={station_number}"
"&help=1&authKey={API_KEY}"

REQ_URL = f"{BASE_URL}{SUB_URL}tm1={start_dt}&tm2={end_dt}&stn={station_number}&help=1&authKey={API_KEY}"

res = requests.get(REQ_URL)
source = res.text.split("\n")
source[65].split()
source = [line.split() for line in source[65:-2]]
df = pd.DataFrame(source)
new_df = df.loc[:,:4]
new_df.columns = ["TM","STN", "WS_AVG", "WS_DAY", "WS_MAX"]

new_df.to_sql(
    "weather_test", con = engine, index=False, if_exists="append")
