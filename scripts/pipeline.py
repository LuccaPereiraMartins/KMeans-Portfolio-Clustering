
import pandas as pd

import processor
import fama_french



def main():

    END_DATE = '2025-01-01'
    TIMEFRAME = 2
    CLUSTERS = 5

    
    try:
        raw_data = pd.read_csv(f'raw_data/ftse250_{int(TIMEFRAME * 12)}months_from_{END_DATE}.csv')
    except:
        # consider reverting to load function
        return 'fail'


    raw_data = pd.read_csv(f'raw_data/ftse250_{int(TIMEFRAME * 12)}months_from_{END_DATE}.csv')
    pre_enriched_data = processor.pre_enrich(raw_data)
    enriched_data = processor.enrich(pre_enriched_data)
    aggregated_data = processor.aggregate_monthly(enriched_data)
    aggregated_w_monthly_data = processor.calculate_monthly_returns(aggregated_data)
    ff_data = fama_french.fama_french(aggregated_w_monthly_data)
    rolling_ff_data = fama_french.rolling_parameters(ff_data)
    final_df = fama_french.append_and_shift(ff_data,rolling_ff_data)

    print(final_df.head())

    # TODO now for the clustering part...


if __name__ == '__main__':
    main()