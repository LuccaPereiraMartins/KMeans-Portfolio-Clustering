
import pandas as pd
import pandas_datareader.data as web

from pipeline import END_DATE, TIMEFRAME

def fama_french(
    df: pd.DataFrame,
) -> None:
    
    ff_data : pd.DataFrame
    ff_data = web.DataReader('F-F_Research_Data_5_Factors_2x3',
                                'famafrench',
                                start='2010')[0].drop('RF', axis=1)

    ff_data.index = ff_data.index.to_timestamp()

    factor_data = ff_data.resample('M').last().div(100)

    factor_data.index.name = 'date'

    factor_data = factor_data.join(df['return_1m'])

    # factor_data = factor_data.join(df['return_1m']).sort_index()

    return factor_data

def main():

    aggregated = pd.read_csv(f'aggregated/ftse250_{int(TIMEFRAME * 12)}months_from_{END_DATE}',
                             index_col='date')
    print(aggregated['return_1m'])

    ff_data = fama_french(df=aggregated)

    print (ff_data)


if __name__ == '__main__':
    main()