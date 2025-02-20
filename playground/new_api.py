

## NOTE: Alpha Vantage free API is limited to only 25 requests per day

from alpha_vantage.timeseries import TimeSeries
from tqdm import tqdm
import pandas as pd
from datetime import datetime

# Configuration: Add your Alpha Vantage API key here
API_KEY = 'LIU4L8PYD2CWS2D3'

# Stock symbols to query (example UK stocks)
stock_symbols = ['TSCO.LON', 'BARC.LON', 'ULVR.LON']

# Date Range
start_date = '2022-01-01'
end_date = '2025-01-01'

# Initialize TimeSeries object with your API key
ts = TimeSeries(key=API_KEY, output_format='pandas')

# Function to get historical data for a single stock
def get_stock_data(symbol):
    try:
        # Get daily adjusted prices
        data, meta_data = ts.get_daily(symbol=symbol, outputsize='full')
        
        # Format the date index
        data.index = pd.to_datetime(data.index)
        
        # Filter data by date range
        filtered_data = data.loc[(data.index >= start_date) & (data.index <= end_date)]
        

        return filtered_data
    except Exception as e:
        print(f"Error fetching data for {symbol}: {e}")
        return None

# Loop through each stock symbol and fetch the data
all_data = {}
for symbol in tqdm(stock_symbols,desc='Fetching market data via API'):
    all_data[symbol] = get_stock_data(symbol)

# Example: Combine all data into a single DataFrame for comparison
combined_df = pd.DataFrame()
for symbol in stock_symbols:
    if all_data[symbol] is not None:
        combined_df[symbol] = all_data[symbol]['4. close']

# Convert to long format
df_long = combined_df.melt(ignore_index=False, var_name="ticker", value_name="adjusted_close")

# Set MultiIndex
df_long = df_long.rename_axis(["date", "ticker"])

print(df_long.head())

# Display combined data
# print("\nCombined Adjusted Close Prices:")
# print(combined_df.head())

# combined_df.to_csv('temp_outputs/3ticker_w_range.csv')


# import pandas as pd
# import matplotlib.pyplot as plt
# df = pd.read_csv('temp_outputs/3ticker_w_range.csv')
# print(df.head())
# plot = df.plot()
# plt.show()




