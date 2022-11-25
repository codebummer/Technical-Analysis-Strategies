from PyQt5.QtWidgets import *
from PyQt5.QAxContainer import *
from PyQt5.QtCore import *
import pandas_datareader.data as web
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import sqlite3
import os, sys, json

#Download a ticker-stock pair dictionary
path = r'D:\myprojects\TradingDB'
if not os.path.exists(path):
    os.mkdir(path)
os.chdir(path)
if not os.path.isfile('stocklist.json'):        
    app = QApplication(sys.argv)
    ocx = QAxWidget('KHOPENAPI.KHOpenAPICtrl.1')
    def login(errcode):
        if errcode == 0:
            print('Logged in successfully')
        else:
            raise Exception('Loggin failed')
        logloop.quit()
    def stockcodes_receive(market=['0','10']):
        codelist = ocx.dynamicCall('GetCodeListByMarket(QString)', market)
        tickers = codelist.split(';')
        stock_list = {'tickerkeys':{}, 'stockkeys':{}}
        for ticker in tickers:
            if ticker == '':
                continue
            else:
                stock = ocx.dynamicCall('GetMasterCodeName(QString)', ticker)
                stock_list['tickerkeys'][ticker] = stock
                stock_list['stockkeys'][stock] = ticker
        with open('stocklist.json', 'w') as file:
            json.dump(stock_list, file)
            print('The ticker-stock pair dictionary saved in stocklist.json under D:\myprojects\TradingDB')
        return stock_list

    ocx.OnEventConnect.connect(login)
    ocx.dynamicCall('CommConnect()')
    logloop = QEventLoop()
    logloop.exec_()
    ticker_stock = stockcodes_receive()
else:
    ticker_stock = {}

#Read all tickers
path = r'D:\myprojects\TradingDB'
if not os.path.exists(path):
    os.mkdir(path)
os.chdir(path)
# #The following three lines read a ticker list from a text file. 
# #However, a json file is read instead, because it is more convinient
# with open('tickers.txt', 'r') as file:
#     tickers = file.read()
# tickers = [ticker.strip(' \'[]\"') for ticker in tickers.split(',')]

#The following two lines read a ticker-stock pair dictionary
#However, they are not used, because it is easier to use the dictionary data
#from the returned value from 'stockcodes_receive()'
if ticker_stock == {}:
    with open('stocklist.json') as file:
        ticker_stock = json.load(file)

#Download daily prices from NAVER
path = r'D:\myprojects\TradingDB\daily'
if not os.path.exists(path):
    os.mkdir(path)
os.chdir(path)
filenames = os.listdir()
if not filenames:
    start = datetime(2021, 1, 1)
    end = datetime.today()
    for ticker in ticker_stock['tickerkeys'].keys():
        df = web.DataReader(ticker, 'naver', start, end)
        df = df.astype('float64')
        with sqlite3.connect(ticker+'.db') as file:
            df.to_sql('Daily_Prices', file)
            print(f'{ticker} saved under D:\myprojects\TradingDB\daily')

#Reierate from here
path = r'D:\myprojects\TradingDB\daily'
if not os.path.exists(path):
    os.mkdir(path)
os.chdir(path)
# filenames = os.listdir()
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
    PERIOD = -100
    if len(df) < -PERIOD:
        PERIOD = -len(df)

    # Conditions by which to screen stocks are created with bool variables which are all capitalized
    MA = True
    mas = [df.MA5, df.MA10, df.MA20, df.MA60, df.MA120]
    ma_compare = [[mas[i], mas[i+1]] for i in range(len(mas)-1)]
    for ma in ma_compare:
        MA = MA and all(ma[0][PERIOD:] > ma[1][PERIOD:])

    DAILYCHANGE = all(-0.03 < df.CloseChangePercent[PERIOD:]) and all(df.CloseChangePercent[PERIOD:] < 0.03)
    for idx in range(PERIOD, 0):
        DAILYCHANGE = DAILYCHANGE and -0.05 < (df.Close.values[idx]/df.Close.values[PERIOD] - 1) < 0.05
    
    ACCUMULATION = True
    ACC_PERIOD = -10
    if len(df) < -ACC_PERIOD:
        ACC_PERIOD = -len(df)
    for idx in range(ACC_PERIOD, 0):
        ACCUMULATION = ACCUMULATION and \
            df.VolChangePercent.values[idx] > 0.5 and 0 < df.CloseChangePercent.values[idx] < 0.02
    
    BANDWIDTH = all(df.Bandwidth[PERIOD:] < 20)
                        
    # Add screen conditions to use in the following if statement          
    # Available conditions are MA, DAILYCHANGE, ACCUMULATION, BANDWIDTH
    if DAILYCHANGE and ACCUMULATION and BANDWIDTH:     
        screened_tickers.append(ticker)
        print(f'{tricker_stripped} selected')
    else:
        print(f'{tricker_stripped} failed')
    

screened_tickers = [stock.strip('.db') for stock in screened_tickers]
screened_stocks = {}
for ticker in screened_tickers:
    screened_stocks[ticker] = ticker_stock['tickerkeys'][ticker]

path = r'D:\myprojects\TradingDB'
if not os.path.exists(path):
    os.mkdir(path)
os.chdir(path)
# with open('screened_stocks.txt', 'w') as file:
#     file.write(str(screened_tickers))
#     print(f'{len(screened_stocks)} stock(s) found. Screen results saved in screened_stocks.txt')
with open('screened_stocks.json', 'w') as file:
    json.dump(screened_stocks, file)
    print(f'{len(screened_stocks.keys())} stock(s) found. Screen results saved in D:\myprojects\TradingDBscreened_stocks.json')
