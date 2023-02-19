import pandas as pd
import numpy as np
import sqlite3
import os, sys, re, json
from datetime import datetime
from tqdm import tqdm
import OpenDartReader
import time
os.chdir(r'C:\Users\omgho\OneDrive\문서\pyprojects')
from DART_key import *

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

    return [table[1].split('_')[0] for table in tables]

def get_ni(corp, date):
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

       
    fins = {'year':[], 'quarter':[]}
            
    for file in tqdm(files):
        with sqlite3.connect(path+'\\'+file) as db:
            query = '''SELECT * FROM sqlite_master WHERE type='table';'''
            tables = db.cursor().execute(query).fetchall()
            tables = [table[1] for table in tables]
            year, quarter = file.strip('.db').split('_')
            table = corp+'_'+changeqtr[quarter]
            fins['year'].append(year)
            fins['quarter'].append(quarter)            
            if table in tables:
                df = pd.read_sql(f'SELECT * FROM [{table}]', db)
                for idx in range(len(df)):
                    value = df.loc[idx,'thstrm_amount'].replace('.','')
                    nonnumeric = re.search('[^-*0-9+.*]',value)
                    if nonnumeric or value == '':
                        fins[df.loc[idx,'account_nm']] = 0.0                        
                    else: 
                        fins[df.loc[idx,'account_nm']] = float(df.loc[idx,'thstrm_amount'])
            else:
                continue
    return fins
        

def get_fins_all():
    '''    
    returns all companies' financial records all vailable in a dictionary
    '''
    
    corps = get_corp_list()

    def _get_single_corp(corp):
        add = {}
        path = r'D:\myprojects\MarketDB\KoreaFins'
        files = os.listdir(path)
        files.sort(reverse=True)
        years = [str(i) for i in range(2022, 2017, -1)]
        quarters = ['Q4', 'Q3', 'Q2', 'Q1']
        changeqtr = {'Q1':'1분기보고서', 'Q2':'반기보고서', 'Q3':'3분기보고서', 'Q4':'사업보고서'}

        add = {'year':[], 'quarter':[]}                
        for file in tqdm(files):
            with sqlite3.connect(path+'\\'+file) as db:
                query = '''SELECT * FROM sqlite_master WHERE type='table';'''
                tables = db.cursor().execute(query).fetchall()
                tables = [table[1] for table in tables]
                year, quarter = file.strip('.db').split('_')
                table = corp+'_'+changeqtr[quarter]
                add['year'].append(year)
                add['quarter'].append(quarter)                
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
        return add

    fins = {}
    for corp in tqdm(corps):
        fins[corp] = _get_single_corp(corp)
        
    with open(r'D:\myprojects\MarketDB\finstats.json', 'w', encoding='utf-8') as file:
        json.dump(fins, file, ensure_ascii=False)
    print('saved financial statements in finanstats.json')
        
    return fins

fins = get_fins_all()


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
        add = {'year':[], 'quarter':[]}
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
                    raw = raw[['account_nm', 'thstrm_amount']]
                    add['year'].append(year)
                    add['quarter'].append(quarter)
                    for idx in range(len(raw)):
                        value = raw.loc[idx,'thstrm_amount'].replace('.','')
                        nonnumeric = re.search('[^-*0-9+.*]',value)
                        if nonnumeric or value == '':
                            add[raw.loc[idx,'account_nm']] = 0.0 
                        else:                           
                            add[raw.loc[idx,'account_nm']] = float(raw.loc[idx,'thstrm_amount'])
        return add

    fins = {}
    for corp in tqdm(corps):    
        fins[corp] = _get_fins_corp(corp)

    with open('finstats.json', 'w', encoding='utf-8') as file:
        json.dump(fins, file, ensure_ascii=False)
    print('saved financial statements in finanstats.json')   
    
    return fins             

fins = get_fins_from_scratch()

with open('finstats.json', 'r', encoding='utf-8') as file:
    fins = json.load(file)
ak = pd.DataFrame(fins['AK홀딩스'])
ak.groupby('year').get_group(2022).groupby('quarter').get_group('Q3')['매출총이익']
