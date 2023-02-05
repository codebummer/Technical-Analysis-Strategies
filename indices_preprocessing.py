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
from pprint import pprint
import sqlite3


def show_info(dic):
    for key, value in dic.items():
        print(key)
        value.info()
        print('\n')

def show_len(dic):
    add = {}
    for key, value in dic.items():
        add[key] = [len(value)]
    return pd.DataFrame(add, index=['length'])        

def date_range(dic):
    add = {}
    for key, value in dic.items():
        add[key] = [value.index[0], value.index[-1]]
    return pd.DataFrame(add, index=['start', 'end'])

def show_columns(dic):
    add = {}
    for key, value in dic.items():
        add[key] = value.columns
    return pprint(add)

def make_isorange(dic, first, last):
    iso_ranged = {}
    for key, value in dic.items():
        firstish = value.index[first <= value.index][0]
        lastish = value.index[value.index <= last][-1]
        iso_ranged[key] = value.loc[firstish:lastish]
    return iso_ranged

def make_isodate(dic, index):
    isodate = pd.DataFrame()
    for key, value in dic.items():
        add = {key:[]}
        for date in index:
            try:
                add[key].append(value.loc[date][0])
            except:
                add[key].append(None)
        isodate = pd.concat([isodate,pd.DataFrame(add, index=index)], axis='columns')
    return isodate

def pull_isocolumn(dic, column):
    add = {}
    for key, value in dic.items():
        try:
            add[key] = pd.DataFrame(value[column].values, index=value.index, columns=[key])
        except:
            add[key] = dic[key]
    return add

def to_sqlite3(df, name):
    with sqlite3.connect(f'./{name}.db') as db:
        df.to_sql('data', db, if_exists='replace', index=False)

# correction required for the following supposedly crawling function from macrotrends.net
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

# get data available from stooq.com
data = {}
start = datetime(1946, 1, 1)
# 'KORM2':'M2SYKR.M' M2 is not vailable
stooq = {'KOSPI':'^KOSPI', 'KOR10Y':'10KRY.B', 
         'S&P500':'^SPX', 'NASDAQ':'^NDQ', 'US10Y':'10USY.B', 'US2Y':'2USY.B'}       

for key, value in tqdm(stooq.items()):
    data[key] = stooq_data(value, start)

# get data available from yahoo finance
yahoo = {'JPY/KRW':'JPYKRW=X', 'USD/KRW':'KRW=X', 'EUR/KRW':'EURKRW=X', 'USD/CNY':'CNY=X'}
for key, value in tqdm(yahoo.items()):
    data[key] = yahoo_data(value, start)

# import CRUDE, DXY, GOLD prices from json files
data['CRUDE'] = crude_data()
data['DXY'] = dxy_data()
data['GOLD'] = gold_data()

# sort values ascending by date
for key in data.keys():
    data[key].sort_index(ascending=True, inplace=True)

# find earlest and latest dates in the dataset
ranges = date_range(data)
first = ranges.loc['start'].max()
last = ranges.loc['end'].min()

# make the dataset ranged in the same date period
iso_ranged = make_isorange(data, first, last)
show_info(iso_ranged)
show_len(iso_ranged)
date_range(iso_ranged)
show_columns(iso_ranged)

# pull only 'Close' values from the dictionary with multiple dataframes and make them one dataframe
iso_columned = pull_isocolumn(iso_ranged, 'Close')
pprint(iso_columned)
show_len(iso_columned)
date_range(iso_columned)
show_columns(iso_columned)

# keep data only with same dates as KOSPI value dates
isodated = make_isodate(iso_columned, iso_columned['KOSPI'].index)
isodated.info()
isodated.describe()
isodated.corr()

sns.heatmap(isodated)
plt.show()

sns.pairplot(isodated)
plt.show()
