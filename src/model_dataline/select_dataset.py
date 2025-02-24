#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sqlalchemy import create_engine
import pandas as pd

def get_dataframe_from_database(database, table_name, all=False, **kwargs):

    ship_id = kwargs.get('ship_id')
    op_index = kwargs.get('op_index')
    section = kwargs.get('section')
    
    # 연결 정보 설정
    username = 'bwms_dba'  # 사용자 이름
    password = '!^admin1234^!'  # 비밀번호
    host = 'signlab.iptime.org'  # 서버 주소
    port = 20002  # 포트

    # SQLAlchemy 엔진 생성
    engine = create_engine(f'mysql+pymysql://{username}:{password}@{host}:{port}/{database}')
    

    if not all: 
        # SQL 쿼리
        query = f"SELECT * FROM `{table_name}` WHERE `SHIP_ID` = '{ship_id}'AND `OP_INDEX` = '{op_index}' AND `SECTION` = '{section}';"
    else:
        # SQL 쿼리
        query = f"SELECT * FROM `{table_name}`"

    # Pandas를 사용하여 데이터 프레임으로 로드
    df = pd.read_sql(query, engine)
    
    return df


