#!/usr/bin/env python3

from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
from_file = __import__('2-from_file').from_file

# Load data from CSV into DataFrame
df = from_file('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')

# Remove 'Weighted_Price' column
df.drop(columns=['Weighted_Price'], inplace=True)

# Rename 'Timestamp' column to 'Date' and convert to datetime
df.rename(columns={'Timestamp': 'Date'}, inplace=True)
df['Date'] = pd.to_datetime(df['Date'], unit='s')

# Index the DataFrame on 'Date'
df.set_index('Date', inplace=True)

# Fill missing values
df['Close'].fillna(method='ffill', inplace=True)
df['High'].fillna(df['Close'], inplace=True)
df['Low'].fillna(df['Close'], inplace=True)
df['Open'].fillna(df['Close'], inplace=True)
df['Volume_(BTC)'].fillna(0, inplace=True)
df['Volume_(Currency)'].fillna(0, inplace=True)

# Resample data for daily intervals from 2017 onwards
df = df['2017':].resample('D').agg({
    'High': 'max',
    'Low': 'min',
    'Open': 'mean',
    'Close': 'mean',
    'Volume_(BTC)': 'sum',
    'Volume_(Currency)': 'sum'
})

# Plotting the data
plt.figure(figsize=(14, 7))
plt.plot(df.index, df['High'], label='High', marker='o')
plt.plot(df.index, df['Low'], label='Low', marker='o')
plt.plot(df.index, df['Open'], label='Open', marker='o')
plt.plot(df.index, df['Close'], label='Close', marker='o')
plt.xlabel('Date')
plt.ylabel('Price')
plt.title('Bitcoin Prices')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
