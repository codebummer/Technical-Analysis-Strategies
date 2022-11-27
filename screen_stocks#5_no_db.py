# from PyQt5.QtWidgets import *
# from PyQt5.QAxContainer import *
# from PyQt5.QtCore import *
import pandas_datareader.data as web
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import sqlite3
import os, sys, json, re

#Global variables
ticker_stock = {}
all_df = {}

#Download or read, if it already exists, a ticker-stock pair dictionary
def get_stocklist():   
    global ticker_stock
    path = r'D:\myprojects\TradingDB'
    if not os.path.exists(path):
        os.mkdir(path)
    os.chdir(path)
    if not os.path.isfile('stocklist.json'):  
        print('\n\nStock list does not exist. Initiating download.') 
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
                print('\nThe ticker-stock pair dictionary saved in stocklist.json under D:\myprojects\TradingDB')
            return stock_list

        ocx.OnEventConnect.connect(login)
        ocx.dynamicCall('CommConnect()')
        logloop = QEventLoop()
        logloop.exec_()
        ticker_stock = stockcodes_receive()
    else:
        with open('stocklist.json') as file:
            ticker_stock = json.load(file)
        print('\n\nStock list already exists. List download will not be implemented.')

#Download daily prices from NAVER
def get_dailydata():
    global ticker_stock, all_df
    path = r'D:\myprojects\TradingDB\daily'
    if not os.path.exists(path):
        os.mkdir(path)
    os.chdir(path)
    filenames = os.listdir()
    found = []
    for filename in filenames:
        add = re.findall('daily_[0-9]*.db', filename)
        if add:
            found.append(add[0])       
    if found:
        print('\nDaily data already exists. Reading daily data from the existing file.')
        with sqlite3.connect(found[-1]) as file:
            for ticker in ticker_stock['tickerkeys'].keys():
                stock = ticker_stock['tickerkeys'][ticker]
                all_df[ticker] = [pd.read_sql(f'SELECT * FROM [Daily_Prices_{ticker}]', file)]  
                print(f'{ticker}, {stock} read')              
        return

    print('\nImplementing daily data download')
    start = datetime(2021, 1, 1)
    end = datetime.today()
    for ticker in ticker_stock['tickerkeys'].keys():
        df = web.DataReader(ticker, 'naver', start, end)
        df = df.astype('float64')
        all_df[ticker] = [df]
        stock = ticker_stock['tickerkeys'][ticker]
        print(f'{ticker}, {stock} downloaded')
    print('\nDownload Completed')

    while True:
        save = input('\nDo you want to save the data? (Yes: 1 , No: 2) ')
        if save == '1':
            print('\nSaving the daily market data.')
            df_name = 'daily'+'_'+str(datetime.today().strftime('%Y%m%d'))+'.db'
            with sqlite3.connect(df_name) as file:
                for ticker in ticker_stock['tickerkeys'].keys():
                    stock = ticker_stock['tickerkeys'][ticker]
                    pd.DataFrame(all_df[ticker][0]).to_sql('Daily_Prices'+'_'+ticker, file)
                    print(f'{ticker}, {stock} saved')
            print(f'\nDaily market data completed. All data saved in D:\myprojects\TradingDB\daily\{df_name}')
            break
        elif save == '2':
            break
        else:
            print('\nPlease choose between 1 and 2')            

#Reierate from here
def screen_stocks(conditions):
    global ticker_stock, all_df
    path = r'D:\myprojects\TradingDB\daily'
    if not os.path.exists(path):
        os.mkdir(path)
    os.chdir(path)

    print('\nInitiating screening processes.\n')

    screened_tickers = []
    for ticker in all_df.keys():
        stock = ticker_stock['tickerkeys'][ticker]
        df = all_df[ticker][0]
        
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


        PERIOD = 100
        if len(df) < PERIOD:
            PERIOD = len(df)

        # Conditions by which to screen stocks are created with bool variables which are all capitalized
        MA = True
        mas = [df.MA5, df.MA10, df.MA20, df.MA60, df.MA120]
        ma_compare = [[mas[i], mas[i+1]] for i in range(len(mas)-1)]
        for ma in ma_compare:
            MA = MA and all(ma[0][-PERIOD:] > ma[1][-PERIOD:])

        DAILYCHANGE = True
        DAILYCHANGERANGE = 0.05
        MAXMINRANGE = 0.05
        DAILYCHANGE = all(-DAILYCHANGERANGE < df.CloseChangePercent[-PERIOD:]) and all(df.CloseChangePercent[-PERIOD:] < DAILYCHANGERANGE)\
                    and -MAXMINRANGE < (df.Close.values[-PERIOD:].max()/df.Close.values[-PERIOD] - 1) < MAXMINRANGE\
                    and -MAXMINRANGE < (df.Close.values[-PERIOD:].min()/df.Close.values[-PERIOD] - 1) < MAXMINRANGE
        for idx in range(-PERIOD, 0):
            DAILYCHANGE = DAILYCHANGE and -MAXMINRANGE < (df.Close.values[idx]/df.Close.values[-PERIOD] - 1) < MAXMINRANGE 
        
        ACCUMULATION = True
        ACC_PERIOD = 5
        if len(df) < ACC_PERIOD:
            ACC_PERIOD = len(df)
        for idx in range(-ACC_PERIOD, 0):
            ACCUMULATION = ACCUMULATION and \
                df.VolChangePercent.values[idx] > 0.3 and 0 < df.CloseChangePercent.values[idx] < 0.025
        
        BANDWIDTH = all(df.Bandwidth[-PERIOD:] < 20)
        
        TRADEPERIOD = 20
        ISTRADE = any(df.Volume[-TRADEPERIOD:] != 0) # 'ISTRADE = not all(df.Volume[-TRADEPERIOD:] != 0)' is the same
            

        CONDITIONS = conditions
        conditions_dict = {'MA':MA, 'DAILYCHANGE':DAILYCHANGE, 'ACCUMULATION':ACCUMULATION, 'BANDWIDTH':BANDWIDTH, 'ISTRADE':ISTRADE}   
        CONDITIONS_PROCESSED = all([conditions_dict[con] for con in CONDITIONS.split(' and ')])
        # if ticker == filenames[0]:
        if ticker == list(all_df.keys())[0]:
            print('Conditions are set for '+CONDITIONS)
        if CONDITIONS_PROCESSED:     
            screened_tickers.append(ticker)
            # print(f'{ticker_stripped}, {stock} selected')        
            print(f'{ticker}, {stock} selected')
        else:
            # print(f'{ticker_stripped}, {stock} failed')
            print(f'{ticker}, {stock} failed')    
    print('\n\nScreen completed')

    # screened_tickers = [stock.strip('.db') for stock in screened_tickers]
    screened_stocks = {}
    for ticker in screened_tickers:
        screened_stocks[ticker] = ticker_stock['tickerkeys'][ticker]

    return screened_stocks

def save_results(conditions, screened_stocks):
    path = r'D:\myprojects\TradingDB'
    if not os.path.exists(path):
        os.mkdir(path)
    os.chdir(path)

    filename_combined = 'screened_stocks'+'_'+conditions+'.json'
    with open(filename_combined, 'w') as file:
        json.dump(screened_stocks, file)
        print(f'{len(screened_stocks.keys())} stock(s) found. Screen results saved in D:\myprojects\TradingDB\{filename_combined}')

def screen_easy(conditions):
    get_stocklist()
    get_dailydata()
    screened_stocks = screen_stocks(conditions)
    save_results(conditions, screened_stocks)

# Add screen conditions to use in the 'screen_easy' function as a string.
# Use 'and' to connect multiple conditions shown below.          
# Available conditions are MA, DAILYCHANGE, ACCUMULATION, BANDWIDTH, ISTRADE
# ie. screen_easy('MA, DAILYCHANGE, ACCUMULATION, BANDWIDTH, ISTRADE')
screen_easy('ACCUMULATION and ISTRADE')
