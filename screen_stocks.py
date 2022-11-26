import pandas_datareader.data as web
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import sqlite3
import os

os.chdir(r'D:\myprojects\TradingDB')

with open('tickers.txt', 'r') as file:
    tickers = file.read()
tickers = [ticker.strip(' \'[]\"') for ticker in tickers.split(',')]

os.chdir(r'D:\myprojects\TradingDB\daily')
start = datetime(2021, 1, 1)
end = datetime.today()
for ticker in tickers:
    df = web.DataReader(ticker, 'naver', start, end)
    df = df.astype('float64')
    with sqlite3.connect(ticker+'.db') as file:
        df.to_sql('Daily_Prices', file)

#Reierate from here
os.chdir(r'D:\myprojects\TradingDB\daily')
filenames = os.listdir()
screened_stocks = []
for ticker in filenames:
    with sqlite3.connect(ticker) as file:
        df = pd.read_sql('SELECT * FROM [Daily_Prices]', file)

    df['MA5'] = df.Close.rolling(window=5).mean()
    df['MA10'] = df.Close.rolling(window=10).mean()
    df['MA20'] = df.Close.rolling(window=20).mean()
    df['MA60'] = df.Close.rolling(window=60).mean()
    df['MA120'] = df.Close.rolling(window=120).mean()
    df['STD'] = df.Close.rolling(window=20).std()
    df['Upper'] = df.MA20 + 2 * df.STD
    df['Lower'] = df.MA20 - 2 * df.STD
    df['PB'] = (df.Close - df.Lower) / (df.Upper - df.Lower)
    df['Bandwidth'] = (df.Upper - df.Lower) / df.MA20 * 100
    df['Diff'] = df.Close.diff(1)
    df['CloseChangePercent'] = df.Close.pct_change(1)
    df['VolChangePercent'] = df.Volume.pct_change(1)

    tricker_stripped = ticker.strip('.db')
    if all(df.Bandwidth < 10) and all(df.MA5 > df.MA10 > df.MA20 > df.MA60 > df.MA120) and any(df.VolChangePercent[-20:] > 0.3):          
# and all(-0.03 < df.CloseChangePercent < 0.03) 
        screened_stocks.append(ticker)
        print(f'{tricker_stripped} selected')
    else:
        print(f'{tricker_stripped} failed')
    

screened_stocks = [stock.strip('.db') for stock in screened_stocks]
os.chdir(r'D:\myprojects\TradingDB')
with open('screened_stocks.txt', 'w') as file:
    file.write(str(screened_stocks))
    print(f'{len(screened_stocks)} stock(s) found. Screen results saved in screened_stocks.txt')
