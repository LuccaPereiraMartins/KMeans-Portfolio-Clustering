
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt


"""
Thoughts:

- Given the clustered data, compare how each clusters performs against the whole FTSE250 index.
- Consider taking the clusters and creating a portfolio for each, using efficient frontier method to optimize portfolio weights
- Plot a graph demonstrating this.
"""


def raw_returns(
        end_date: str = '2025-01-01',
        timeframe: int = 2  
) -> pd.DataFrame:
    
    start_date = pd.to_datetime(end_date)-pd.DateOffset(365*timeframe)  

    ftse_250 = yf.download(tickers='^FTMC',
                    start=start_date,
                    end=end_date)
    
    return_ftse_250 = ftse_250[['Close']].pct_change().fillna(value=float(0))

    return return_ftse_250


def portfolio_returns(
        data,
        portfolio_number: int = 3,
) -> pd.DataFrame:
    
    data = pd.read_csv('temp_clustered.csv', index_col=('date','ticker'))

    pf_cluster : pd.DataFrame = data[data['cluster'] == portfolio_number].copy()
    dates = pf_cluster.index.get_level_values('date').unique().tolist()

    final = pd.Series()

    for date in dates:

        pd_date = pd.to_datetime(str(date))
        # get the first and last business day for that month
        first_bday = pd_date + pd.offsets.BMonthBegin(0)
        last_bday = pd_date + pd.offsets.BMonthEnd(0)
        # get the list of tickers in that cluster for that month
        tickers = pf_cluster.loc[date].index.tolist()
        # get the raw data for those tickers (this could be done outside of the loop)
        prices_for_month = yf.download(tickers=tickers,
                    start=first_bday,
                    end=last_bday)
        
        # find the underliers individual daily percentage change
        filtered_data = prices_for_month['Close'][tickers].pct_change().fillna(value=float(0))
        # here is where we can introduce weighting and further portfolio optimization
        # for now just weight every stock equally hence take arithmetic mean
        final = pd.concat([final, np.mean(filtered_data, axis=1)])

    return final


def create_plot(
        raw_returns: np.array,
):
    

    # plot the FTSE250 returns as a benchmark
    plt.plot(raw_returns.cumsum(), color='blue', label='FTSE 250')

    # Add labels and legend
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Percentage Change (%)', fontsize=12)
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
    plt.plot(portfolio_returns.cumsum(), color=(1 - portfolio_number/20, 0.5 + portfolio_number/20, 0 + portfolio_number/20), alpha = 0.75, label=f'Portfolio #{portfolio_number}')

def main():

    pd.set_option("display.max_rows", 50)
    
    # save the results to avoid multiple querying
    if True:

        _raw_returns = raw_returns()
        _raw_returns.to_csv('raw_data/FTSE250_ticker_daily_change.csv')

        data = pd.read_csv('temp_clustered.csv', index_col=('date','ticker'))
        portfolio_3_returns = portfolio_returns(data=data,portfolio_number=3)
        portfolio_3_returns.to_csv('processed_data/portfolio_3_daily_change.csv')

    # TODO load in the data from the above csv files instead of calling yfinance API every time

    if True:
        # plot the benchmark index performance and set up graph
        create_plot(
            raw_returns=_raw_returns,
        )

    # plot a specific portfolio
    if True:
        plot_pf_return(
            portfolio_returns=portfolio_returns(data=data,portfolio_number=3),
            portfolio_number=3,
        )

    # plot all portfolios
    if False:
        for pf in [0,1,2,3]:
            plot_pf_return(
                portfolio_returns=portfolio_returns(data, pf),
                portfolio_number=pf
            )

    if True:
        plt.legend(fontsize=10, loc='upper left')
        plt.show()



if __name__ == '__main__':
    main()