
import sqlalchemy as sqa
import pandas as pd
import yfinance as yf







def connect_to_db(
        db_url: str = "postgresql+psycopg2://postgres:password@127.0.0.1:5432/postgres"
):
    
    engine = sqa.create_engine(db_url)
    return engine


def bulk_upload(
        engine: sqa.Engine
):

    # database connection details
    db_url = "postgresql+psycopg2://postgres:password@127.0.0.1:5432/postgres"

    # create SQLAlchemy engine
    engine = sqa.create_engine(db_url)

    # drop existing table if exists then create table with right params
    with engine.connect() as conn:
        with open(r"scripts\ingest_data\ingest_data.sql", "r") as file:
            sql_query = file.read()
            conn.execute(sqa.text(sql_query))
            conn.commit()

    # TODO replace the below with data retrieved from yf api call
    # this might require some tidying between the api call and upload, similar to what's in load_data.py

    # load in data to df from csv
    initial_data = pd.read_csv(r'raw_data\ftse250_24months_from_2025-01-01.csv')
    initial_data.columns = ['date', 'ticker', 'adj_close', 'close', 'high', 'low', 'open', 'volume']

    # upload DataFrame to PostgreSQL, handling duplicate rows thanks to primary key
    initial_data.to_sql('market_data', engine, schema='raw', if_exists='append', index=False)

    # check the upload was successful
    results = pd.read_sql('select * from raw.market_data', engine)
    assert not results.isnull().values.any(), "Upload failed: NULL values found"

    engine.dispose()


def incremental_upload():
    pass


# TODO consider converting this to a class and methods to make connecting and closing easier

class database():

    def __init__(self, db_url):
        self.engine: sqa.Engine = sqa.create_engine(db_url)

    def conn(self):
        self.engine.connect()
    
    def disconnect(self):
        self.engine.dispose()



def main():
    pass


if __name__ == '__main__':
    main()