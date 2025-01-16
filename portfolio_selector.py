
import math

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
) -> np.array:
    
    start_date = pd.to_datetime(end_date)-pd.DateOffset(365*timeframe)  

    ftse_250 = yf.download(tickers='^FTMC',
                    start=start_date,
                    end=end_date)
    
    return_ftse_250 = ftse_250[['Adj Close']].pct_change().dropna()
    return_ftse_250 = np.array(return_ftse_250).cumsum()

    return return_ftse_250


def portfolio_returns(
        clustered_data,
):
    
    _portfolio_returns = clustered_data
    return _portfolio_returns


def plot_returns(
        raw_returns: np.array,
        portfolio_returns: np.array = None,
):
    
    plt.plot(raw_returns)
    plt.plot(portfolio_returns)
    plt.show()


def main():

    pd.set_option("display.max_rows", 50)

    # plot_raw_returns(raw_returns())

    # read in the processed data with multi-index
    data = pd.read_csv('temp_clustered.csv', index_col=('date','ticker'))
    
    pf_cluster_3 : pd.DataFrame = data[data['cluster'] == 3].copy()
    dates = pf_cluster_3.index.get_level_values('date').unique().tolist()
    dict_pf_cluster_3 : dict = {}
    returns_pf_3: list[float] = []
    daily_returns_pf3 : list[float] = []
    for date in dates:

        first_bday = pd.to_datetime(str(date)) + pd.offsets.BMonthBegin(0)
        last_bday = pd.to_datetime(str(date)) + pd.offsets.BMonthEnd(0)
        # get the list of tickers in that cluster for that month
        tickers = pf_cluster_3.loc[date].index.tolist()
        # get the raw data for those tickers
        prices_for_month = yf.download(tickers=tickers,
                    start=first_bday,
                    end=last_bday)
        # get the percentage return for each of those tickers for that month
        pct_change_by_ticker = []
        daily_pct_change_by_ticker = []
        for index, ticker in enumerate(tickers):

            # daily frequency
            indv_pct_change = prices_for_month['Adj Close'][str(ticker)].pct_change().dropna()
            if not any(math.isnan(x) for x in indv_pct_change) \
                and not indv_pct_change.empty \
                and not all(float(x) == float(0) for x in indv_pct_change):
                daily_pct_change_by_ticker.append(list(indv_pct_change.values))

            if not prices_for_month['Adj Close'][str(ticker)].empty:
                first = prices_for_month['Adj Close'][str(ticker)].values[0]
                last = prices_for_month['Adj Close'][str(ticker)].values[-1]
                pct_change = float((last - first) / first)
                # only append values that aren't nan
                if not math.isnan(pct_change): pct_change_by_ticker.append(pct_change)

        # here is where we can introduce weighting and further portfolio optimization
        # for now just weight every stock equally hence take arithmetic mean
        weighted_change = np.mean(pct_change_by_ticker)
        returns_pf_3.append(weighted_change)


        averaged_daily_pct_change_by_ticker = np.mean(np.array(daily_pct_change_by_ticker), axis=1)
        daily_returns_pf3.extend(averaged_daily_pct_change_by_ticker)
        
        # append all this information to a possibly useless dictionary
        dict_pf_cluster_3[date] = {
            'ticker' : tickers,
            'pct_change_by_ticker': pct_change_by_ticker,
            'avg_return' : weighted_change,
            }

    pf3_cum = np.array(returns_pf_3).cumsum()
    daily_pf3_cum = np.array(daily_returns_pf3).cumsum()
    _raw_returns = raw_returns()
    
    plt.plot(pf3_cum)

    plot_returns(
        raw_returns = _raw_returns,
        portfolio_returns = daily_pf3_cum,
    )



if __name__ == '__main__':
    main()