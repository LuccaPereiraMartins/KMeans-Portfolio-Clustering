import sqlalchemy as sqa
import pandas as pd




def connect_to_db(
        db_url: str = "postgresql+psycopg2://postgres:password@127.0.0.1:5432/postgres"
):
    
    """
    Connect to the postgres database, by keeping seperate we avoid having to reconnect several times
    """

    engine = sqa.create_engine(db_url)
    return engine


def bulk_upload(
        engine: sqa.Engine
) -> None:
    
    """
    TEMP
    
    Take the csv file which holds the raw data we are interested in and upload it to postgres
    """

    # drop existing table if exists then create table with right params
    with engine.connect() as conn:
        with open(r"scripts\ingest_data\ingest_data.sql", "r") as file:
            sql_query = file.read()
            conn.execute(sqa.text(sql_query))
            conn.commit()

    # TODO replace the below with data retrieved from yf or alpha vantage api call
    # this might require some tidying between the api call and upload, similar to what's in load_data.py

    # load in data to df from csv
    initial_data = pd.read_csv(r'raw_data\ftse250_24months_from_2025-01-01.csv')
    initial_data.columns = ['date', 'ticker', 'adj_close', 'close', 'high', 'low', 'open', 'volume']

    # upload DataFrame to PostgreSQL, handling duplicate rows thanks to primary key
    initial_data.to_sql('market_data', engine, schema='raw', if_exists='append', index=False)

    # check the upload was successful
    results = pd.read_sql('select * from raw.market_data limit 10', engine)
    assert not results.isnull().values.any(), "Upload failed: NULL values found"

    engine.dispose()


def incremental_upload():
    pass




class Engine():

    def __init__(self, db_url):
        self.engine: sqa.Engine = sqa.create_engine(db_url)

    def connect(self):
        return self.engine.connect()
    
    def disconnect(self):
        self.engine.dispose()



def main():
    db_url = "postgresql+psycopg2://postgres:password@127.0.0.1:5432/postgres"
    engine_instance = Engine(db_url)
    
    with engine_instance.connect() as conn:
        # run test query
        results = pd.read_sql('select * from raw.market_data limit 10', conn)
        print(results)
    
    engine_instance.disconnect()


if __name__ == '__main__':
    main()