
import pandas as pd
import numpy as np
import pandas_ta



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

    """Financial enriching manipulation steps."""
    df['garman_klass_vol'] = ((np.log(df['high'])-np.log(df['low']))**2)/2-(2*np.log(2)-1)*((np.log(df['adj close'])-np.log(df['open']))**2)
    df['rsi'] = pandas_ta.rsi(df['adj close'], length=20)
    bb = pandas_ta.bbands(close=np.log1p(df['adj close']), length=20)
    df['bb_low'] = bb.iloc[:,2]
    df['bb_mid'] = bb.iloc[:,3]
    df['bb_high'] = bb.iloc[:,4]
    df['atr'] = _compute_atr(df=df)
    df['macd'] = _compute_macd(df=df)
    df['dollar_volume'] = (df['adj close']*df['volume'])/1e6
    return df


def _compute_atr(
        df: pd.DataFrame
) -> pd.Series:
    """Compute normalized ATR indicator."""
    atr = pandas_ta.atr(high=df['high'],
                        low=df['low'],
                        close=df['close'],
                        length=14)
    return atr.sub(atr.mean()).div(atr.std())


def _compute_macd(
        df: pd.DataFrame
) -> pd.Series:
    """Compute normalized MACD indicator."""
    macd = pandas_ta.macd(close=df['adj close'], length=20).iloc[:,0]
    return macd.sub(macd.mean()).div(macd.std())


def aggregate_monthly(df: pd.DataFrame) -> pd.DataFrame:

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
            group['dollar_volume'].resample('ME').mean().to_frame('dollar_volume'),
            group[training_columns].resample('ME').last()
        ], axis=1)
    )

    # reset the indexes
    data = data.reset_index(level=1,drop=False).reset_index(drop=True)  

    return data


def calculate_monthly_returns(
    df: pd.DataFrame
) -> pd.DataFrame:

    outlier_cutoff = 0.005

    lags = [1, 2, 3, 6, 9, 12]

    for lag in lags:

        df[f'return_{lag}m'] = (df['adj close']
                              .pct_change(lag,fill_method=None)
                              .pipe(lambda x: x.clip(lower=x.quantile(outlier_cutoff),
                                                     upper=x.quantile(1-outlier_cutoff)))
                              .add(1)
                              .pow(1/lag)
                              .sub(1))
    
    return df.dropna()


def main():
    pass


if __name__ == '__main__':
    main()