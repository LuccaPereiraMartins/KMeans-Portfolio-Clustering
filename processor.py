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


def pre_enrich(
    df: pd.DataFrame,
    ) -> pd.DataFrame:

    """non-financial pre-enriching manipulation steps

    Returns:
        _type_: _description_
    """
    
    # Turn column headers to lowercase for ease
    df.columns = df.columns.str.lower()

    # Ensure MultiIndex with proper names
    if not isinstance(df.index, pd.MultiIndex):
        df = df.set_index(['date', 'ticker'])
    
    # Assign new names for multi-index
    df.index.names = ['date', 'ticker']

    # Convert the `date` index level to just the date part
    df.index = df.index.set_levels([
        pd.to_datetime(df.index.levels[0]).normalize(),
        df.index.levels[1]
    ])

    return df


def enrich(
    df: pd.DataFrame,
    ) -> pd.DataFrame:


    df['garman_klass_vol'] = ((np.log(df['high'])-np.log(df['low']))**2)/2-(2*np.log(2)-1)*((np.log(df['adj close'])-np.log(df['open']))**2)

    df['rsi'] = df.groupby(level=1)['adj close'].transform(lambda x: pandas_ta.rsi(close=x, length=20))

    df['bb_low'] = df.groupby(level=1)['adj close'].transform(lambda x: pandas_ta.bbands(close=np.log1p(x), length=20).iloc[:,0])
                                                            
    df['bb_mid'] = df.groupby(level=1)['adj close'].transform(lambda x: pandas_ta.bbands(close=np.log1p(x), length=20).iloc[:,1])
                                                            
    df['bb_high'] = df.groupby(level=1)['adj close'].transform(lambda x: pandas_ta.bbands(close=np.log1p(x), length=20).iloc[:,2])

    df['atr'] = df.groupby(level=1, group_keys=False).apply(compute_atr)

    df['macd'] = df.groupby(level=1, group_keys=False)['adj close'].apply(compute_macd)

    df['dollar_volume'] = (df['adj close']*df['volume'])/1e6

    return df


def compute_atr(stock_data):
    atr = pandas_ta.atr(high=stock_data['high'],
                        low=stock_data['low'],
                        close=stock_data['close'],
                        length=14)
    return atr.sub(atr.mean()).div(atr.std())


def compute_macd(close):
    macd = pandas_ta.macd(close=close, length=20).iloc[:,0]
    return macd.sub(macd.mean()).div(macd.std())


def main():
    
    data = pd.read_csv('raw_data/ftse250_24months_from_2025-01-01')

    pre_enriched = pre_enrich(data)

    pre_enriched.style
    print(pre_enriched.head(10))

    # enriched = enrich(pre_enrich)

    # print(enriched.head())


if __name__ == '__main__':
    main()