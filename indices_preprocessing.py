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

def show_info(df):
    for key, value in df.items():
        print(key)
        value.info()
        print('\n')

def macrotrends_data():
    from bs4 import BeautifulSoup
    import selenium
    import requests

    url = 'https://www.macrotrends.net/assets/php/chart_iframe_comp.php?id=1333&url=historical-gold-prices-100-year-chart'
    with requests.get(url) as page:
        source = BeautifulSoup(page.text)

    scripts = source.find_all('script', type='text/javascript')
    script = scripts[5]
    script.find('var', 'originalData')

def crude_data():
    with open('./oil_price_history.json','r') as file:
        oil = json.loads(file.read())

    crude = {'Date':[], 'Close':[]}
    for daily in oil:
        crude['Date'].append(datetime.strptime(daily['date'],'%Y-%m-%d'))
        crude['Close'].append(float(daily['close'])) 

    return pd.DataFrame(crude['Close'], index=crude['Date'], columns=['CRUDE'])

def dxy_data():
    with open('./DXY.json', 'r') as file:
        dxy = json.loads(file.read())
    
    DXY = {'Date':[], 'Close':[]}
    for daily in dxy:
        DXY['Date'].append(datetime.strptime(daily['date'], '%Y-%m-%d'))
        DXY['Close'].append(float(daily['close']))
    return pd.DataFrame(DXY['Close'], index=DXY['Date'], columns=['DXY'])

def gold_data():
    with open('./gold.json', 'r') as file:
        gold = json.loads(file.read())
    
    GOLD = {'Date':[], 'Close':[]}
    for daily in gold:
        GOLD['Date'].append(datetime.strptime(daily['date'], '%Y-%m-%d'))
        GOLD['Close'].append(float(daily['close']))
    return pd.DataFrame(GOLD['Close'], index=GOLD['Date'], columns=['GOLD'])


def stooq_data(ticker, start):
    return pdr.get_data_stooq(ticker, start, datetime.today())

def yahoo_data(ticker, start):
    return pdr.get_data_yahoo(ticker, start, datetime.today())

crude = crude_data()
dxy = dxy_data()
gold = gold_data()


start = datetime(1946, 1, 1)
# 'KORM2':'M2SYKR.M' M2 is not vailable
stooq = {'KOSPI':'^KOSPI', 'KOR10Y':'10KRY.B', 
         'S&P500':'^SPX', 'NASDAQ':'^NDQ', 'US10Y':'10USY.B', 'US2Y':'2USY.B'}       
data = {}
for key, value in tqdm(stooq.items()):
    data[key] = stooq_data(value, start)

yahoo = {'JPY/KRW':'JPYKRW=X', 'USD/KRW':'KRW=X', 'EUR/KRW':'EURKRW=X', 'USD/CNY':'CNY=X'}
for key, value in tqdm(yahoo.items()):
    data[key] = yahoo_data(value, start)

data['DXY'] = dxy
data['GOLD'] = gold
data['CRUDE'] = crude

# sort
for key in data.keys():
    data[key].sort_index(ascending=True, inplace=True)

show_info(data)

for key, value in data.items():
    print(key, ': ', len(value))

for key, value in data.items():
    print(key, ': ', value.index[0], '  ', value.index[-1])
