
import pandas as pd
import pandas_datareader.data as web
import statsmodels.api as sm
from statsmodels.regression.rolling import RollingOLS


def fama_french(
    df: pd.DataFrame,
) -> pd.DataFrame:
    
    ff_data : pd.DataFrame
    ff_data = web.DataReader('F-F_Research_Data_5_Factors_2x3',
                                'famafrench',
                                start='2020')[0].drop('RF', axis=1)

    # convert columns and index to lower for ease
    ff_data.columns = ff_data.columns.str.lower()
    ff_data.index.name = ff_data.index.name.lower()

    # ensure both indexes are datetimes ahead of merging
    ff_data.index = pd.to_datetime(ff_data.index.to_timestamp())
    df.set_index('date', inplace=True)

    # resampling to last day of month and divide by 100
    ff_data = ff_data.resample('ME').last().div(100)

    # merge the two datasources with a left outer merge
    # this keeps all the columns from df and copies across
    # the ff_data into each date, for each ticker
    df = df.merge(ff_data,  how='left', left_index=True, right_index=True).dropna()

    return df


def rolling_parameters(
        df: pd.DataFrame,
) -> pd.DataFrame:

    # Convert DataFrame columns to NumPy arrays
    endog = df['return_1m'].values
    exog = sm.add_constant(df[['mkt-rf', 'smb', 'hml', 'rmw', 'cma']].values)

    # Set the rolling window size
    window = min(24, len(endog))  # Ensure window size doesn't exceed the data length

    # Initialize RollingOLS and fit
    rolling_model = RollingOLS(endog=endog, exog=exog, window=window, min_nobs=exog.shape[1] + 1)
    results = rolling_model.fit(params_only=True)

    # Convert results.params (numpy.ndarray) to pandas DataFrame for easier handling
    param_columns = ['mkt-rf', 'smb', 'hml', 'rmw', 'cma']
    params_df = pd.DataFrame(results.params[:, 1:], columns=param_columns, index=df.index)

    return params_df


def append_and_shift(
        ff_data: pd.DataFrame,
        rolling: pd.DataFrame
) -> pd.DataFrame:
    

    # append then shift the rolling data to be month+1 from rest of data
    ff_data[['mkt-rf', 'smb', 'hml', 'rmw', 'cma']] = rolling
    ff_data[['mkt-rf', 'smb', 'hml', 'rmw', 'cma']] = (
    ff_data.groupby('ticker')[['mkt-rf', 'smb', 'hml', 'rmw', 'cma']]
    .transform(lambda x: x.shift()))
    
    final_df = ff_data.dropna().copy()
    final_df = final_df.drop(labels=['adj close'], axis=1)
    final_df.index = final_df.index.to_period('M')

    # Set 'ticker' as part of the multi-index
    final_df = final_df.set_index('ticker', append=True)
    
    return final_df


def main():
    pass


if __name__ == '__main__':
    main()