import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt


def raw_returns(
        end_date: str = '2025-01-01',
        timeframe: int = 2  
) -> pd.DataFrame:
    """
    When needed, retrieve the FTSE250 returns for a given period of time from yfinance.

    Args:
        end_date (str, optional). Defaults to '2025-01-01'.
        timeframe (int, optional): In years. Defaults to 2.

    Returns:
        pd.DataFrame: percentage change returns of FTSE250
    """
    start_date = pd.to_datetime(end_date)-pd.DateOffset(365*timeframe)  

    ftse_250 = yf.download(tickers='^FTMC',
                    start=start_date,
                    end=end_date)
    
    return_ftse_250 = ftse_250[['Close']].pct_change().fillna(value=float(0))

    return return_ftse_250


def portfolio_returns(
    data: pd.DataFrame,
    portfolio_number: int = 3,
    raw_data: pd.DataFrame = None
) -> pd.Series:
    """
    Given a portfolio (cluster) number, calculate the daily percentage change of the portfolio.
    Use the raw data extracted, find the tickers composing the portfolio in a given month.
    Room to introduce a weighting function to optimize the portfolio further, for now use even weights.

    Args:
        data (pd.DataFrame): Clustered data with 'cluster' column.
        portfolio_number (int): Portfolio/cluster number.
        raw_data (pd.DataFrame): Raw price data (optional, recommended for efficiency).

    Returns:
        pd.Series: A series of the daily percentage change of the portfolio
    """
    pf_cluster: pd.DataFrame = data[data['cluster'] == portfolio_number].copy()
    dates = pf_cluster.index.get_level_values('date').unique().tolist()

    final = pd.Series(dtype=float)

    if raw_data is None:
        raw_data = pd.read_csv('raw_data/ftse250_24months_from_2025-01-01.csv',header=0)
    
    raw_data['Date'] = pd.to_datetime(raw_data['Date']).dt.date

    for date in dates:
        first_bday, last_bday = _get_first_last_bd(str(date))
        tickers = pf_cluster.loc[date].index.tolist()
        df_trim = raw_data[(raw_data['Date'] >= first_bday) & (raw_data['Date'] <= last_bday)].copy()
        df_pivot = df_trim.pivot(index='Date', columns='Ticker', values='Close')
        pct_change_data = df_pivot.pct_change(fill_method=None).fillna(value=float(0))
        filtered_data = pct_change_data[tickers]
        monthly_average = np.mean(filtered_data, axis=1)
        final = pd.concat([final, monthly_average])

    return final


def _get_first_last_bd(date: str):
    """Given a date, get the first and last business day of that month."""
    pd_date = pd.to_datetime(str(date))
    first_bday = (pd_date + pd.offsets.BMonthBegin(0)).date()
    last_bday = (pd_date + pd.offsets.BMonthEnd(0)).date()
    return first_bday, last_bday


def create_plot(
        x_data,
        y_data,
):
    
    # plot the FTSE250 returns as a benchmark
    plt.plot(x_data, y_data, color='blue', label='FTSE 250')

    # Add labels and legend
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Cumulative Percentage Change \'00s(%)', fontsize=12)
    plt.title('Performance Comparison', fontsize=14)
    plt.legend(fontsize=10, loc='upper left')

    # Enhance the grid and layout
    plt.grid(alpha=0.5, axis='both')
    plt.tight_layout()


def plot_pf_return(
        portfolio_returns,
        portfolio_number: int = 3,
        daily_change: bool = False
):

    if daily_change:        
        plt.plot(portfolio_returns, color=(0.5 - portfolio_number/100, 0.5 - portfolio_number/100, 0.5 - portfolio_number/100),
                  linestyle='dashed', alpha=0.5, label=f'Daily Change of Portfolio #{portfolio_number}')
    plt.plot(portfolio_returns.cumsum(), color=(1 - portfolio_number/10, 0.5 + portfolio_number/10, 0 + portfolio_number/10), alpha = 0.75, label=f'Portfolio #{portfolio_number+1}')
    plt.legend(fontsize=10, loc='upper left')


def main():
    test_final = portfolio_returns(data=pd.read_csv('processed_data/clustered_data.csv',header=0),portfolio_number=3)
    create_plot(x_data=test_final.index,y_data=test_final)
    plot_pf_return(test_final,portfolio_number=3)
    plt.show()