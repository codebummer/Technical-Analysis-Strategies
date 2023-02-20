import pandas as pd
import numpy as np
import sqlite3
import os, sys, re, json
from datetime import datetime
from tqdm import tqdm
import OpenDartReader
import xmltodict
import time
os.chdir(r'D:\myprojects\MarketDB')
from DART_key import *

def get_tickers():
    os.chdir(r'D:\myprojects\MarketDB\corpCode')
    with open('CORPCODE.xml', encoding='utf-8') as file:
        corps = xmltodict.parse(file.read())
        corps = corps['result']['list']

    tickers = {'ticker':{}, 'stock':{}}
    for corp in corps:
        if corp['stock_code'] != None:        
            tickers['ticker'][corp['stock_code']] = corp['corp_name']
            tickers['stock'][corp['corp_name']] = corp['stock_code']
            
    with open('ticker.json', 'w', encoding='utf-8') as file:
        json.dump(tickers, file)
        
    return tickers


def change_filenames():
    '''
    change the format of the file names downloaded from DART and saved in the local storage
    '''
    quarters = {'1분기보고서':'Q1', '반기보고서':'Q2', '3분기보고서':'Q3', '사업보고서':'Q4'}
    os.chdir(r'D:\myprojects\MarketDB\KoreaFins')
    files = os.listdir()
    for file in tqdm(files):
        _, year, quarter = file.strip('.db').split('_')
        os.rename(file, year+'_'+quarters[quarter]+'.db')

def get_corp_list():
    path = r'D:\myprojects\MarketDB\KoreaFins'
    files = os.listdir(path)
    files.sort(reverse=True)
    with sqlite3.connect(path+'\\'+files[-1]) as db:
        query = '''SELECT * FROM sqlite_master WHERE type='table';'''
        tables = db.cursor().execute(query).fetchall()
        tables = [table[1].split('_')[0] for table in tables]
        tables.sort()
    return tables

def get_shares(stock):
    dart = OpenDartReader(DART_KEY)
    today = datetime.today()
    tickers = get_tickers()
    for year in range(today.year, today.year-6, -1):
        for quarter in ['11011','11014','11012','11013']:
            shares = dart.report(tickers['stock'][stock], '주식총수', year, quarter)
            if len(shares) != 0:
                if any(shares['isu_stock_totqy'] != '-'):
                    return shares

def get_close_prices():
    os.chdir(r'D:\myprojects\MarketDB')
    files = os.listdir()
    if 'close_prices.csv' not in files:
        with open('ticker.json', encoding='utf-8') as file:
            tickers = json.load(file)

        prices = pd.DataFrame()
        for ticker, stock in tqdm(tickers['ticker'].items()):
            try:
                series = pdr.DataReader(ticker, 'naver')['Close']
                series.name = stock
                prices = pd.concat([prices, series], axis='columns')
            except:
                continue
        prices.index.name = 'Date'
        for stock in tqdm(prices):
            if all(prices[stock]) == np.nan:
                prices.drop(stock, inplace=True)
        prices.to_csv('close_prices.csv', encoding='utf-8-sig')
        
        return prices
    
    else:
        return pd.read_csv('close_prices.csv', parse_dates=['Date'], index_col=0)


def get_single_ni(corp, date):
    '''
    corp: company name
    date: datetime.datetime value
    returns Net Income of the company in the quarter
    '''
    quarters = ['Q1', 'Q2' ,'Q3', 'Q4']
    if 1 <= date.month <= 3:
        quarter = quarters[0]
    elif 4 <= date.month <= 6:
        quarter = quarters[1]
    elif 7 <= date.month <= 9:
        quarter = quarters[2]
    else:
        quarter = quarters[3]

    path = r'D:\myprojects\MarketDB\KoreaFins'
    file = str(date.year)+'_'+quarter+'.db'
    with sqlite3.connect(path+'\\'+file) as db:
        table = corp+'_'+quarter
        df = pd.read_sql(f'SELECT * FROM {table}', db)
        return int(df.loc[df['account_nm'].str.contains('순이익')].loc[df['account_detail'].str.contains('연결재무제표'),'thstrm_amount'].iloc[0])

def get_ni():
    os.chdir(r'D:\myprojects\MarketDB')
    with open('finstats.json', encoding='utf-8') as file:
        fins = json.load(file)

    corps = get_corp_list()

    ni = pd.DataFrame.from_dict(fins[corps[0]])[['year','quarter']]
    for corp in tqdm(corps):
        df = pd.DataFrame.from_dict(fins[corp])
        candidates = df.loc[:,df.columns.str.contains('당기순이익')]
        candidates.columns = candidates.columns.sort_values()
        for col in candidates.columns:
            if all(candidates[col])==0:
                continue
            else:
                add = candidates[col]
                add.name = corp
                ni = pd.concat([ni,add], axis='columns')
                break
    return ni


def get_fins_corp(corp):
    '''
    corp: company name
    returns a single company's financial records all vailable in a dictionary
    '''
    path = r'D:\myprojects\MarketDB\KoreaFins'
    files = os.listdir(path)
    files.sort(reverse=True)
    years = [str(i) for i in range(2022, 2017, -1)]
    quarters = ['Q4', 'Q3', 'Q2', 'Q1']
    changeqtr = {'Q1':'1분기보고서', 'Q2':'반기보고서', 'Q3':'3분기보고서', 'Q4':'사업보고서'}

       
    fins = []            
    for file in tqdm(files):
        with sqlite3.connect(path+'\\'+file) as db:
            add = {}
            query = '''SELECT * FROM sqlite_master WHERE type='table';'''
            tables = db.cursor().execute(query).fetchall()
            tables = [table[1] for table in tables]
            year, quarter = file.strip('.db').split('_')
            table = corp+'_'+changeqtr[quarter]
            add['year'] = year
            add['quarter'] = quarter            
            if table in tables:
                df = pd.read_sql(f'SELECT * FROM [{table}]', db)
                for idx in range(len(df)):
                    value = df.loc[idx,'thstrm_amount'].replace('.','')
                    nonnumeric = re.search('[^-*0-9+.*]',value)
                    if nonnumeric or value == '':
                        add[df.loc[idx,'account_nm']] = 0.0                        
                    else: 
                        add[df.loc[idx,'account_nm']] = float(df.loc[idx,'thstrm_amount'])
            else:
                continue
            
            fins.append(add)
            
    return fins        

def get_fins_all():
    '''    
    returns all companies' financial records all vailable in a dictionary
    '''
    os.chdir(r'D:\myprojects\MarketDB')
    files = os.listdir()
    if 'finstats.json' not in files:
        corps = get_corp_list()

        def _get_single_corp(corp):
            add = {}
            path = r'D:\myprojects\MarketDB\KoreaFins'
            files = os.listdir(path)
            files.sort(reverse=True)
            years = [str(i) for i in range(2022, 2017, -1)]
            quarters = ['Q4', 'Q3', 'Q2', 'Q1']
            changeqtr = {'Q1':'1분기보고서', 'Q2':'반기보고서', 'Q3':'3분기보고서', 'Q4':'사업보고서'}

            records = []                
            for file in tqdm(files):
                with sqlite3.connect(path+'\\'+file) as db:
                    add = {}
                    query = '''SELECT * FROM sqlite_master WHERE type='table';'''
                    tables = db.cursor().execute(query).fetchall()
                    tables = [table[1] for table in tables]
                    year, quarter = file.strip('.db').split('_')
                    table = corp+'_'+changeqtr[quarter]
                    add['year'] = year
                    add['quarter'] = quarter
                    if table in tables:
                        df = pd.read_sql(f'SELECT * FROM [{table}]', db)
                        for idx in range(len(df)):
                            value = df.loc[idx,'thstrm_amount'].replace('.','')
                            nonnumeric = re.search('[^-*0-9+.*]',value)
                            if nonnumeric or value == '':
                                add[df.loc[idx,'account_nm']] = 0.0 
                            else:                        
                                add[df.loc[idx,'account_nm']] = float(df.loc[idx,'thstrm_amount'])
                    else:
                        continue
                    
                    records.append(add)
                    
            return records

        fins = {}
        for corp in tqdm(corps):
            fins[corp] = _get_single_corp(corp)
            
        with open(r'D:\myprojects\MarketDB\finstats.json', 'w', encoding='utf-8') as file:
            json.dump(fins, file, ensure_ascii=False)
        print('saved financial statements in finanstats.json')
            
        return fins
    
    else:
        with open('finstats.json', 'r', encoding='utf-8') as file:
            return json.load(file)


def get_fins_from_scratch():
    with sqlite3.connect(r'C:\Users\omgho\OneDrive\문서\pyprojects\fins_2022_Q3.db') as db:
        query = '''SELECT * FROM sqlite_master WHERE type='table';'''
        tables = db.cursor().execute(query).fetchall()
        corps = [table[1].split('_')[0] for table in tables]
        corps.sort()
        corps = corps[:700]


    dart = OpenDartReader(DART_KEY)
    years = [year for year in range(2022, 2017, -1)]
    quarters = {'11013':'Q1', '11012':'Q2', '11014':'Q3', '11011':'Q4'}

    def _get_fins_corp(corp):
        records = []
        for year in years:
            for code, quarter in quarters.items():            
                try:
                    raw = dart.finstate_all(corp, year, code)
                    # without time.sleep(0.5), an error will occur
                    time.sleep(0.5)
                except:
                    time.sleep(0.5)
                    continue
                if len(raw) != 0:
                    add={}
                    raw = raw[['account_nm', 'thstrm_amount']]
                    add['year'] = year
                    add['quarter'] = quarter
                    for idx in range(len(raw)):
                        value = raw.loc[idx,'thstrm_amount'].replace('.','')
                        nonnumeric = re.search('[^-*0-9+.*]',value)
                        if nonnumeric or value == '':
                            add[raw.loc[idx,'account_nm']] = 0.0 
                        else:                           
                            add[raw.loc[idx,'account_nm']] = float(raw.loc[idx,'thstrm_amount'])
                            
                    records.append(add)
                    
        return records

    fins = {}
    for corp in tqdm(corps):    
        fins[corp] = _get_fins_corp(corp)

    with open('finstats.json', 'w', encoding='utf-8') as file:
        json.dump(fins, file, ensure_ascii=False)
    print('saved financial statements in finanstats.json')   
    
    return fins  

# fins = get_fins_from_scratch()

# with open('finstats.json', 'r', encoding='utf-8') as file:
#     fins = json.load(file)
# ak = pd.DataFrame.from_dict(fins['AK홀딩스'])
# ak.loc[:,ak.columns.str.contains('당기순이익')]
