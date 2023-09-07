import os
import sys
from src.mlproject.exception import CustomException
from src.mlproject.logger import logging
import pandas as pd
from dotenv import load_dotenv
import pyodbc

load_dotenv()

server = os.getenv("server")
db = os.getenv("db")
user = os.getenv("user")
password = os.getenv("password")
# Depending on your SQL Server setup and the ODBC driver installed, the driver name might be different.
driver= os.getenv("driver")

SQL_QUERY = """
select * from AccessLevel_1;
"""

def read_sql_data():
    logging.info("Reading SQL database started")
    try:
        # defining connection string      
        connectionString = f'DRIVER={driver};SERVER={server};DATABASE={db};UID={user};PWD={password}'
        connection = pyodbc.connect(connectionString)
        
        # logging the succesful connection
        logging.info("Connection Established",connection)

        df=pd.read_sql_query(SQL_QUERY, connection)
        # print(df.head())
        
        # Close the connection
        # connection.close()

        return df


    except Exception as ex:
        raise CustomException(ex)