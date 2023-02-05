import pandas as pd
import numpy as np
import pandas_datareader.data as pdr
import yfinance as yf
yf.pdr_override()
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json
from tqdm import tqdm

with open('./oil_price_history.json','r') as file:
    oil = json.loads(file.read())

crude = {'Date':[], 'Close':[]}
for daily in oil:
    crude['Date'].append(datetime.strptime(daily['date'],'%Y-%m-%d'))
    crude['Close'].append(float(daily['close']))       
crude = pd.DataFrame(crude['Close'], index=crude['Date'])


def get_data(ticker, start):
    return pdr.get_data_stooq(ticker, start, datetime.today())['Close']

start = datetime(1946, 1, 1)
lists = {'KOSPI':'^KOSPI', 'KOR10Y':'10KRY.B', 'KORM2':'M2SYKR.M', 
         'CNY/KRW':'CNYKRW', 'JPY/KRW':'JPYKRW', 'USD/KRW':'USDKRW', 'DXY':'DX.F',
         'S&P500':'^SPX', 'NASDAQ':'^NDQ', 'US10Y':'10USY.B', 'US2Y':'2USY.B', 
         'GOLD':'XAUUSD'}

data = {}
for key in lists.keys():
    data[key] = pd.DataFrame()
for key in tqdm(lists):
    data[key] = get_data('^KOSPI', start).sort_index(ascending=False, inplace=True)
