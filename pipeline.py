
import pandas as pd

import processor
from load_data import load
from kmeans_model import cluster, pipeline_cluster, plot_clusters


"""
Thoughts:

take a month's worth of data and use it to group similar stocks 
based on the 20 features calculated
repeat for each month then select a portfolio based on
Efficient Frontier max sharpe ratio optimization
compare against the wider FTSE250 index

move all the processes into the pipeline and tidy the local data
"""

def main():

    END_DATE = '2025-01-01'
    TIMEFRAME = 2
    CLUSTERS = 5

    
    try:
        raw_data = pd.read_csv(f'raw_data/ftse250_{int(TIMEFRAME * 12)}months_from_{END_DATE}')
    except:
        raw_data = load(
            end_date=END_DATE,
            timeframe=TIMEFRAME,
            )

    pre_enriched_data = processor.pre_enrich(raw_data)
    enriched_data = processor.enrich(pre_enriched_data)
    aggregated_data = processor.aggregate_monthly(enriched_data)
    aggregated_w_monthly_data = processor.calculate_monthly_returns(aggregated_data)

    # pass aggregated through fama-french script
    # clear issues with indexes, worth going through and changing this in functions

    ff_data = None
    rolling_ff_data = None
    clustered_data = None


    data = pd.read_csv('final_df.csv')
    data = data.set_index(['date','ticker'])

    print(data.loc[('2024-01')])


if __name__ == '__main__':
    main()