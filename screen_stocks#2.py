import pandas_datareader.data as web
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import sqlite3
import os, json

#Read all tickers
os.chdir(r'D:\myprojects\TradingDB')
with open('tickers.txt', 'r') as file:
    tickers = file.read()
tickers = [ticker.strip(' \'[]\"') for ticker in tickers.split(',')]
with open('stocklist.json') as file:
    ticker_stock = json.load(file)

#Download daily prices from NAVER
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
screened_tickers = []
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
    PERIOD = -200

    MA = True
    mas = [df.MA5, df.MA10, df.MA20, df.MA60, df.MA120]
    ma_compare = [[mas[i], mas[i+1]] for i in range(len(mas)-1)]
    for ma in ma_compare:
        MA = MA and all(ma[0][PERIOD:] > ma[1][PERIOD:])

    CLOSECHANGE = all(-0.03 < df.CloseChangePercent[PERIOD:]) and all(df.CloseChangePercent[PERIOD:] < 0.03)
    for idx in range(PERIOD, 0):
        CLOSECHANGE = CLOSECHANGE and -0.03 < (df.Close.values[idx]/df.Close.values[PERIOD] - 1) < 0.03
    # if MA and CLOSECHANGE and all(df.Bandwidth[PERIOD-100:] < 10) and any(df.VolChangePercent[PERIOD:] > 0.3):          
    if CLOSECHANGE and all(df.Bandwidth[PERIOD:] < 20) and any(df.VolChangePercent[-20:] > 0.3):          

        screened_tickers.append(ticker)
        print(f'{tricker_stripped} selected')
    else:
        print(f'{tricker_stripped} failed')
    

screened_tickers = [stock.strip('.db') for stock in screened_tickers]
screened_stocks = {}
for ticker in screened_tickers:
    screened_stocks[ticker] = ticker_stock['tickerkeys'][ticker]

os.chdir(r'D:\myprojects\TradingDB')
# with open('screened_stocks.txt', 'w') as file:
#     file.write(str(screened_tickers))
#     print(f'{len(screened_stocks)} stock(s) found. Screen results saved in screened_stocks.txt')
with open('screened_stocks.json', 'w') as file:
    json.dump(screened_stocks, file)
    print(f'{len(screened_stocks.keys())} stock(s) found. Screen results saved in screened_stocks.json')
