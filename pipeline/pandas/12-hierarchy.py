#!/usr/bin/env python3

import pandas as pd
from_file = __import__('2-from_file').from_file

df1 = from_file('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')
df2 = from_file('bitstampUSD_1-min_data_2012-01-01_to-2020-04-22.csv', ',')

# Set index to 'Timestamp' for both DataFrames
df1 = df1.set_index('Timestamp')
df2 = df2.set_index('Timestamp')

# Concatenate the specified timestamps and rearrange MultiIndex levels
df_concat = pd.concat([df2.loc[1417411980:1417417980], df1.loc[1417411980:1417417980]], keys=['bitstamp', 'coinbase'])

# Swap levels in MultiIndex to have 'Timestamp' as the first level
df_concat = df_concat.swaplevel()

# Sort the index to display rows in chronological order
df_concat = df_concat.sort_index()

print(df_concat)
