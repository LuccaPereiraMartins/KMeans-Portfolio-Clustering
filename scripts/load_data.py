
import pandas as pd
import yfinance as yf


# FTSE250 loading and manipulating

def load(
        end_date: str = '2025-01-01',
        timeframe: int = 2
) -> pd.DataFrame:
    
    # Extract a list of current FTSE250 tickers for end_date and timeframe
    # Dependant on website layout and column names
    ftse250 = pd.read_html('https://en.wikipedia.org/wiki/FTSE_250_Index')[3]
    tickers_ftse250 = ftse250['Ticker'].str.replace('.', '-').unique().tolist()

    # Create dataframe from yfinance api for each constituent in index
    # Use a daily frequency over timeframe from end_date

    start_date = pd.to_datetime(end_date)-pd.DateOffset(365*timeframe)

    df =  yf.download(tickers=tickers_ftse250,
                    start=start_date,
                    end=end_date).stack()

    # Create a csv from the data to inspect later without having to redownload
    df.to_csv(f'raw_data/ftse250_{int(timeframe * 12)}months_from_{end_date}')

    return df




# TODO complete the below function

def load_from_db(
    start_date: str,
    end_date: str,      
):
    
    df = pd.read_sql(
        sql = 
        f"""
        select * from raw.market_data
        where 
            date between {start_date} and {end_date}
        order by date asc, ticker asc
        """
        ,
        con=None
    )


def main():
    pass


if __name__ == '__main__':
    main()