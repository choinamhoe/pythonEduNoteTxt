#create_engine 형태로 바로 사용가능
from sqlalchemy import create_engine, text
#sqlalchemy.create_engine 꺼내서 사용해야 함
#import sqlalchemy
#pandas.DataFrame 너무 길어서 쓰기 불편
import pandas
#pd.DataFrame 짧게 사용이 가능
import pandas as pd


import bcrypt

user_id = "root"
password = "test"
host = "localhost"
port = 3307

db_info = f"mysql+pymysql://{user_id}:{password}@{host}:{port}"
engine = create_engine(
    db_info,connect_args={}
    )

with engine.connect() as conn:
    result = conn.execute(text("show databases;"))
    result = result.fetchall()
    
query = " CREATE USER if not exists 'dev'@'%' IDENTIFIED BY 'dev'; "

with engine.connect() as conn:
    result = conn.execute(text(query))

query = "SELECT user, host from mysql.user; "
with engine.connect() as conn:
    result = conn.execute(text(query))
    result = result.fetchall()
    
result

query = " GRANT ALL PRIVILEGES ON *.* TO 'dev'@'%'; "
with engine.connect() as conn:
    conn.execute(text(query))
    conn.execute(text("FLUSH PRIVILEGES;"))
    res = conn.execute(text("SHOW GRANTS FOR 'dev'@'%'; "))
    print(res.fetchall())

#auth 데이터베이스 만들기
query = "CREATE DATABASE IF NOT EXISTS auth;"
with engine.connect() as conn:
    conn.execute(text(query))
    res = conn.execute(text("show databases;"))
    print(res.fetchall())


query = "USE auth;"
with engine.connect() as conn:
    conn.execute(text(query))    

query = """
CREATE TABLE IF NOT EXISTS users
(id INT AUTO_INCREMENT PRIMARY KEY,
 userid varchar(255) NOT NULL UNIQUE KEY,
 userName varchar(255) NOT NULL,
 password VARCHAR(255) NOT NULL
 )
"""

with engine.connect() as conn:
    conn.execute(text(query))

with engine.begin() as conn:
    conn.execute(text("TRUNCATE TABLE users"))

userid = "choinamhoe"
username = "최남회"
password = "mook176!"

hashed = bcrypt.hashpw(password.encode(),bcrypt.gensalt()).decode("utf-8")
"""
유저생성
"""
query = f"""
INSERT INTO auth.users
(userid, userName, password)
VALUES( '{userid}', '{username}', '{hashed}');
"""

with engine.begin() as conn:
    conn.execute(text(query))

query = "SELECT * FROM users"
with engine.connect() as conn:
    res = conn.execute(text(query))
    print(res.fetchall())
    
"""
유저검증
"""
query = f"""
SELECT password FROM users WHERE userid = '{userid}';
"""

with engine.connect() as conn:
    res = conn.execute(text(query))
    hashed_password = res.fetchone()[0]
bcrypt.checkpw(password.encode(), hashed_password.encode())
bcrypt.checkpw('aaa'.encode(), hashed_password.encode())


#select * from table where True ; 주게 되면 모든 테이블을 반환
#select * from table;과 같은 결과 반환
# OR '1'='1' 이라고 조건을 주었기 때문에 WHERE 비밀번호검증 or '1'='1' 이라고 됨
userid  = "abc' OR '1'='1'"
query = f"""
SELECT password FROM users WHERE userid = '{userid}';
"""

with engine.connect() as conn:
    res = conn.execute(text(query))
    hashed_password = res.fetchone()[0]
hashed_password


# SQL 인젝션 방지
userid  = "abc' OR '1'='1'"
userid = "choinamhoe"
query = """
SELECT password FROM users WHERE userid = :userid;
"""

with engine.connect() as conn:
    res = conn.execute(text(query),{"userid":userid})
    hashed_password = res.fetchone()[0]
hashed_password

#auth 데이터 베이스 다시 연결.
#이유는 아래 csv파일을  auth 데이터베이스에 넣기 위해 다시 연결처리
user_id = "root"
password = "test"
host = "localhost"
port = 3307
db_info = f"mysql+pymysql://{user_id}:{password}@{host}:{port}/auth"
engine = create_engine(
    db_info,connect_args={}
    )

import pandas as pd
df = pd.read_csv("E:/최남회/파이썬개발에대한파일모음/기상철 기상자료개발포털/META_관측지점정보_20250808122943.csv",encoding="cp949")

df.columns
# iloc : 컴퓨터 입장으로 0~입력된 숫자만큼 행,열의 형태로 추출
# df.iloc[:4,:4] : 0부터 시작한 행에서 4행,0부터 시작한 열에서 4열까지 추출
df.iloc[:4,:4]
df.loc[:"종료일"]
df.loc[:,["종료일","시작일"]]
new_df = df.loc[:,["지점","위도","경도"]]
new_df.columns = ["location","lat","lon"]
new_df
new_df.to_sql("location_df",con = engine, index=False,if_exists="append")
new_df2 = pd.read_sql("select * from location_df", con=engine)
new_df2
