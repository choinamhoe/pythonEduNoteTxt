#create_engine 형태로 바로 사용가능
from sqlalchemy import create_engine, text
#sqlalchemy.create_engine 꺼내서 사용해야 함
#import sqlalchemy
#pandas.DataFrame 너무 길어서 쓰기 불편
import pandas
#pd.DataFrame 짧게 사용이 가능
import pandas as pd
import requests


import bcrypt

user_id = "root"
password = "test"
host = "localhost"
port = 3307
db_info = f"mysql+pymysql://{user_id}:{password}@{host}:{port}/auth"
engine = create_engine(
    db_info,connect_args={}
    )
#https://apihub.kma.go.kr/api/typ01/url/kma_sfctm2.php?tm=202211300900&stn=0&help=1&authKey=K66mUw9zTByuplMPczwchQ
API_KEY = "K66mUw9zTByuplMPczwchQ"
BASE_URL = "https://apihub.kma.go.kr/api/typ01/url/"
SUB_URL = "kma_sfctm3.php?"
start_dt = "2015-12-11 01:00"
end_dt = "2015-12-12 23:00"

start_dt = pd.to_datetime(start_dt).strftime("%Y%m%d%H%M")
end_dt = pd.to_datetime(end_dt).strftime("%Y%m%d%H%M")
station_number = 108

REQ_URL = f"{BASE_URL}{SUB_URL}tm1={start_dt}&tm2={end_dt}&stn={station_number}&help=1&authKey={API_KEY}"
res = requests.get(REQ_URL)
res.status_code
source = res.text.split("\n")
source[65].split()
source = [line.split(" ") for line in source[65:-2]]
df = pd.DataFrame(source)
new_df = df.loc[:,:4]
new_df.columns = ["TM","STN","WS_AVG","WS_DAY","WS_MAX"]
new_df.to_sql(
    "weather_test", con = engine, index=False, if_exists="append")
new_df2 = pd.read_sql("select * from weather_test", con=engine)
new_df2
