
import pandas as pd

# TODO move processes from processor and load_data into here
# set any required globals
END_DATE = '2025-01-01'
TIMEFRAME = 2

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
    
    data = pd.read_csv('final_df.csv')
    data = data.set_index(['date','ticker'])

    print(data.loc[('2024-01')])


if __name__ == '__main__':
    main()