
import pandas as pd
import numpy as np
import pandas_ta
from pandasgui import show as gui_show

from pipeline import END_DATE, TIMEFRAME

def pre_enrich(
    df: pd.DataFrame,
    ) -> pd.DataFrame:

    """non-financial pre-enriching manipulation steps

    Inputs:
        raw_data [pd.DataFrame]

    Returns:
        pre_enrich data [pd.DataFrame]
    """
    
    # Turn column headers to lowercase for ease
    df.columns = df.columns.str.lower()

    # Convert the date column to only the date part
    df['date'] = pd.to_datetime(df['date']).dt.date

    return df


def enrich(
    df: pd.DataFrame,
    ) -> pd.DataFrame:

    """financial enriching manipulation steps

    Inputs:
        pre_enrich data [pd.DataFrame]

    Returns:
        enriched data [pd.DataFrame]
    """

    df['garman_klass_vol'] = ((np.log(df['high'])-np.log(df['low']))**2)/2-(2*np.log(2)-1)*((np.log(df['adj close'])-np.log(df['open']))**2)

    df['rsi'] = pandas_ta.rsi(df['adj close'], length=20)

    df['bb_low'] = df['adj close'].transform(lambda x: pandas_ta.bbands(close=pd.Series(np.log1p(x)), length=20).iloc[:,2])
                                                            
    df['bb_mid'] = df['adj close'].transform(lambda x: pandas_ta.bbands(close=pd.Series(np.log1p(x)), length=20).iloc[:,3])
                                                            
    df['bb_high'] = df['adj close'].transform(lambda x: pandas_ta.bbands(close=pd.Series(np.log1p(x)), length=20).iloc[:,4])

    df['atr'] = compute_atr(df=df)

    df['macd'] = compute_macd(df=df)

    df['dollar_volume'] = (df['adj close']*df['volume'])/1e6

    return df


def compute_atr(
        df: pd.DataFrame
) -> pd.Series:
    
    atr = pandas_ta.atr(high=df['high'],
                        low=df['low'],
                        close=df['close'],
                        length=14)
    
    return atr.sub(atr.mean()).div(atr.std())


def compute_macd(
        df: pd.DataFrame
) -> pd.Series:

    macd = pandas_ta.macd(close=df['adj close'], length=20).iloc[:,0]

    return macd.sub(macd.mean()).div(macd.std())


def aggregate_monthly(df: pd.DataFrame) -> pd.DataFrame:

    # TODO this can probably be cleaned up in the pre_enrich steps
    # Set the date column as the index
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)

    # Select the columns we want to keep
    training_columns = [
        'ticker',
        'adj close',
        'atr',
        'bb_high',
        'bb_low',    
        'bb_mid',    
        'garman_klass_vol',    
        'macd',    
        'rsi',
    ]

    # Group by ticker and resample monthly for each ticker
    # dollar_volume -> monthly mean
    # training_columns -> last value of the month

    data = df.groupby('ticker').apply(
        lambda group: pd.concat([
            group['dollar_volume'].resample('M').mean().to_frame('dollar_volume'),
            group[training_columns].resample('M').last()
        ], axis=1)
    ).reset_index(level=0, drop=True)  # Reset the ticker index

    return data


def calculate_monthly_returns(
    df: pd.DataFrame
) -> pd.DataFrame:

    outlier_cutoff = 0.005

    lags = [1, 2, 3, 6, 9, 12]

    for lag in lags:

        df[f'return_{lag}m'] = (df['adj close']
                              .pct_change(lag)
                              .pipe(lambda x: x.clip(lower=x.quantile(outlier_cutoff),
                                                     upper=x.quantile(1-outlier_cutoff)))
                              .add(1)
                              .pow(1/lag)
                              .sub(1))
    return df


def main():

    
    if False:
        data = pd.read_csv('raw_data/ftse250_24months_from_2025-01-01')
        
        pre_enriched = pre_enrich(data)

        enriched = enrich(pre_enriched)

        enriched.to_csv(f'enriched_data/ftse250_{int(TIMEFRAME * 12)}months_from_{END_DATE}')

    if True:
        enriched = pd.read_csv(f'enriched_data/ftse250_{int(TIMEFRAME * 12)}months_from_{END_DATE}')
        
        aggregated = aggregate_monthly(enriched)

        aggregated = calculate_monthly_returns(aggregated).dropna()

        aggregated.to_csv(f'aggregated/ftse250_{int(TIMEFRAME * 12)}months_from_{END_DATE}')
        
        
        print(aggregated)
        # gui_show(aggregated)

if __name__ == '__main__':
    main()