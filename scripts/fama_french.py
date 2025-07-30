
import pandas as pd
import pandas_datareader.data as web
import statsmodels.api as sm
from statsmodels.regression.rolling import RollingOLS


def fama_french(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge Fama-French 5-factor data with input DataFrame by date.
    Args:
        df (pd.DataFrame): Input DataFrame with 'date' column.
    Returns:
        pd.DataFrame: DataFrame merged with Fama-French factors.
    """
    ff_data = web.DataReader('F-F_Research_Data_5_Factors_2x3', 'famafrench', start='2020')[0].drop('RF', axis=1)
    ff_data.columns = ff_data.columns.str.lower()
    ff_data.index = pd.to_datetime(ff_data.index.to_timestamp())
    df = df.copy()
    df.set_index('date', inplace=True)
    ff_data = ff_data.resample('ME').last().div(100)
    df = df.merge(ff_data, how='left', left_index=True, right_index=True).dropna()
    return df


def rolling_parameters(
        df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute rolling Fama-French factor loadings using RollingOLS.
    Args:
        df (pd.DataFrame): DataFrame with returns and factors.
    Returns:
        pd.DataFrame: DataFrame of rolling factor loadings.
    """
    endog = df['return_1m'].values
    exog = sm.add_constant(df[['mkt-rf', 'smb', 'hml', 'rmw', 'cma']].values)
    window = min(24, len(endog))
    rolling_model = RollingOLS(endog=endog, exog=exog, window=window, min_nobs=exog.shape[1] + 1)
    results = rolling_model.fit(params_only=True)
    param_columns = ['mkt-rf', 'smb', 'hml', 'rmw', 'cma']
    params_df = pd.DataFrame(results.params[:, 1:], columns=param_columns, index=df.index)
    return params_df


def append_and_shift(
        ff_data: pd.DataFrame,
        rolling: pd.DataFrame
) -> pd.DataFrame:
    """
    Append and shift rolling factor loadings to align with next month.
    Args:
        ff_data (pd.DataFrame): DataFrame with Fama-French factors.
        rolling (pd.DataFrame): DataFrame of rolling factor loadings.
    Returns:
        pd.DataFrame: Final DataFrame with shifted factors and multi-index.
    """
    ff_data = ff_data.copy()
    ff_data[['mkt-rf', 'smb', 'hml', 'rmw', 'cma']] = rolling
    ff_data[['mkt-rf', 'smb', 'hml', 'rmw', 'cma']] = (
        ff_data.groupby('ticker')[['mkt-rf', 'smb', 'hml', 'rmw', 'cma']]
        .transform(lambda x: x.shift())
    )
    final_df = ff_data.dropna().copy()
    final_df = final_df.drop(labels=['adj close'], axis=1)
    final_df.index = final_df.index.to_period('M')
    final_df = final_df.set_index('ticker', append=True)
    return final_df