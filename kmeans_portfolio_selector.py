from statsmodels.regression.rolling import RollingOLS
import pandas_datareader.data as web
import matplotlib.pyplot as plt
import statsmodels.api as sm
import pandas as pd
import numpy as np
import datetime as dt
import yfinance as yf
import pandas_ta
import warnings



# FTSE250 loading and manipulating

def load(
        end_date: str = '2024-09-10',
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
    df.to_csv(f'ftse250_24months_from_{end_date}')

# Manipulate df, calculate features and technical indicators for each stock

def enrich_df(df: pd.DataFrame) -> pd.DataFrame:

    # Assign new names for multi-index df
    df.index.names = ['date', 'ticker']

    # Assign column headers to strings (easier for manipulation later)
    df.columns = df.columns.str.lower()

    # Convert the `date` index to just the date part
    df.index = df.index.set_levels([pd.to_datetime(df.index.levels[0]).date, df.index.levels[1]])

    df['garman_klass_vol'] = ((np.log(df['high'])-np.log(df['low']))**2)/2-(2*np.log(2)-1)*((np.log(df['adj close'])-np.log(df['open']))**2)

    df['rsi'] = df.groupby(level=1)['adj close'].transform(lambda x: pandas_ta.rsi(close=x, length=20))

    df['bb_low'] = df.groupby(level=1)['adj close'].transform(lambda x: pandas_ta.bbands(close=np.log1p(x), length=20).iloc[:,0])
                                                            
    df['bb_mid'] = df.groupby(level=1)['adj close'].transform(lambda x: pandas_ta.bbands(close=np.log1p(x), length=20).iloc[:,1])
                                                            
    df['bb_high'] = df.groupby(level=1)['adj close'].transform(lambda x: pandas_ta.bbands(close=np.log1p(x), length=20).iloc[:,2])

    def compute_atr(stock_data):
        atr = pandas_ta.atr(high=stock_data['high'],
                            low=stock_data['low'],
                            close=stock_data['close'],
                            length=14)
        return atr.sub(atr.mean()).div(atr.std())

    df['atr'] = df.groupby(level=1, group_keys=False).apply(compute_atr)

    def compute_macd(close):
        macd = pandas_ta.macd(close=close, length=20).iloc[:,0]
        return macd.sub(macd.mean()).div(macd.std())

    df['macd'] = df.groupby(level=1, group_keys=False)['adj close'].apply(compute_macd)

    df['dollar_volume'] = (df['adj close']*df['volume'])/1e6

    return df




def main():

    if False:
        load() 

    df = pd.read_csv('ftse250_24months_from_2024-09-10')
    df = enrich_df(df)

    print(df) 

if __name__ == "__main__":
    main()