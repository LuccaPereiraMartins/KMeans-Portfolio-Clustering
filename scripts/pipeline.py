import os

import pandas as pd
import matplotlib.pyplot as plt

import load_data
import processor
import fama_french
import kmeans_processor
import portfolio_selector



def main():

    END_DATE = '2025-01-01'
    TIMEFRAME = 2
    CLUSTERS = 4

    
    # load the raw data
    filepath = f'raw_data/ftse250_{int(TIMEFRAME * 12)}months_from_{END_DATE}.csv'
    if os.path.exists(filepath):
        try:
            raw_data = pd.read_csv(filepath)
        except FileNotFoundError:
            raw_data = load_data.load(end_date=END_DATE, timeframe=TIMEFRAME)

    # processor
    pre_enriched_data = processor.pre_enrich(raw_data)
    enriched_data = processor.enrich(pre_enriched_data)
    aggregated_data = processor.aggregate_monthly(enriched_data)
    aggregated_w_monthly_data = processor.calculate_monthly_returns(aggregated_data)

    # fama_french
    ff_data = fama_french.fama_french(aggregated_w_monthly_data)
    rolling_ff_data = fama_french.rolling_parameters(ff_data)
    final_df = fama_french.append_and_shift(ff_data,rolling_ff_data)

    # clustering (pipeline_cluster can take an argument 'clustering_model' if necessary)
    clustered_data = kmeans_processor.pipeline_cluster(final_df,clusters=CLUSTERS)
    # clustered_data.to_csv('processed_data/clustered_data.csv')

    # plotting
    _raw_returns = pd.read_csv('raw_data/FTSE250_ticker_daily_change.csv')
    _raw_returns['Date'] = pd.to_datetime(_raw_returns['Date'])

    portfolio_selector.create_plot(
        x_data=_raw_returns['Date'],
        y_data=_raw_returns['Close'].cumsum())
    
    for pf in range(CLUSTERS):

        pf_returns = portfolio_selector.portfolio_returns(
            data=clustered_data,
            portfolio_number=pf
        )
        portfolio_selector.plot_pf_return(
            portfolio_returns=pf_returns,
            portfolio_number=pf,
            daily_change=False
        )


if __name__ == '__main__':
    main()
    plt.show()