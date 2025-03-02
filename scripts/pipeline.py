
import pandas as pd
import matplotlib.pyplot as plt

import processor
import fama_french
import kmeans_processor
import portfolio_selector



def main():

    END_DATE = '2025-01-01'
    TIMEFRAME = 2
    CLUSTERS = 5

    
    # try:
    #     raw_data = pd.read_csv(f'raw_data/ftse250_{int(TIMEFRAME * 12)}months_from_{END_DATE}.csv')
    # except:
    #     # consider reverting to load function
    #     return 'fail'


    # run through clustering block
    if False:
        raw_data = pd.read_csv(f'raw_data/ftse250_{int(TIMEFRAME * 12)}months_from_{END_DATE}.csv')

        pre_enriched_data = processor.pre_enrich(raw_data)
        enriched_data = processor.enrich(pre_enriched_data)
        aggregated_data = processor.aggregate_monthly(enriched_data)
        aggregated_w_monthly_data = processor.calculate_monthly_returns(aggregated_data)

        ff_data = fama_french.fama_french(aggregated_w_monthly_data)
        rolling_ff_data = fama_french.rolling_parameters(ff_data)
        final_df = fama_french.append_and_shift(ff_data,rolling_ff_data)

        clustered_data = kmeans_processor.pipeline_cluster(final_df)

    # now compare against the index
    # TODO portfolio selector script could do with lots of refining

    # load from yfinance:
    if False:
        _raw_returns = portfolio_selector.raw_returns()
    # load from local csv
    if True:
        _raw_returns = pd.read_csv('raw_data/FTSE250_ticker_daily_change.csv')
        _raw_returns['Date'] = pd.to_datetime(_raw_returns['Date'])

    portfolio_selector.create_plot(
        x_data=_raw_returns['Date'],
        y_data=_raw_returns['Close'].cumsum())
    
    # plot the portfolio returns
    # for pf in CLUSTERS:

    # TODO fix the below
    # TODO find a way to avoid using yfinance API every time we evaluate a portfolio

    # pf = 3
    # pf_returns = portfolio_selector.portfolio_returns(
    #     data=clustered_data,
    #     portfolio_number=pf
    # )
    # portfolio_selector.plot_pf_return(
    #     portfolio_returns=pf_returns,
    #     portfolio_number=pf,
    #     daily_change=False
    # )

    temp_pf_returns = pd.read_csv(
        'processed_data\portfolio_3_daily_change.csv',
        parse_dates=['Date']
    )

    portfolio_selector.create_plot(
        x_data=temp_pf_returns['Date'],
        y_data=temp_pf_returns['Close'].cumsum())

    plt.show()


if __name__ == '__main__':
    main()