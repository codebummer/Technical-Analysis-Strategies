import pandas as pd
import numpy as np
import sqlite3
import os, sys, re
from datetime import datetime
from tqdm import tqdm

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
    os.chdir(r'D:\myprojects\MarketDB\KoreaFins')
    path = r'D:\myprojects\MarketDB\KoreaFins'
    years = [str(i) for i in range(2022, 2017, -1)]
    quarters = ['Q4', 'Q3', 'Q2', 'Q1']
    changeqtr = {'Q1':'1분기보고서', 'Q2':'반기보고서', 'Q3':'3분기보고서', 'Q4':'사업보고서'}
   
    fins = {}
    for year in years:
        for quarter in quarters:
            fins[year+'_'+quarter] = {}
            
    for file in tqdm(files):
        with sqlite3.connect(file) as db:
            year, quarter = file.strip('.db').split('_')
            table = corp+'_'+changeqtr[quarter]
            df = pd.read_sql(f'SELECT * FROM {table}', db)
            for idx in range(len(df)):
                value = df.loc[idx,'thstrm_amount'].replace('.','')
                nonnumeric = re.search('[^-*0-9+.*]',value)
                if nonnumeric or value == '':
                    fins[year+'_'+quarter][df.loc[idx,'account_nm']] = 0.0
                else:
                    fins[year+'_'+quarter][df.loc[idx,'account_nm']] = float(df.loc[idx,'thstrm_amount'])
    return fins
        
def get_fins_all():
    '''    
    returns all companies' financial records all vailable in a dictionary
    '''
    os.chdir(r'D:\myprojects\MarketDB\KoreaFins')
    path = r'D:\myprojects\MarketDB\KoreaFins'
    years = [str(i) for i in range(2022, 2017, -1)]
    quarters = ['Q4', 'Q3', 'Q2', 'Q1']
    changeqtr = {'Q1':'1분기보고서', 'Q2':'반기보고서', 'Q3':'3분기보고서', 'Q4':'사업보고서'}
    
    corps = get_corp_list()

    fins = {}
    for corp in corps:
        add = {}
        for year in years:
            for quarter in quarters:
                add[year+'_'+quarter] = {}
                
        for file in tqdm(files):
            with sqlite3.connect(file) as db:
                query = '''SELECT * FROM sqlite_master WHERE type='table';'''
                tables = db.cursor().execute(query).fetchall()
                year, quarter = file.strip('.db').split('_')
                table = corp+'_'+changeqtr[quarter]
                if table in tables:
                    df = pd.read_sql(f'SELECT * FROM {table}', db)
                    for idx in range(len(df)):
                        value = df.loc[idx,'thstrm_amount'].replace('.','')
                        nonnumeric = re.search('[^-*0-9+.*]',value)
                        if nonnumeric or value == '':
                            add[year+'_'+quarter][df.loc[idx,'account_nm']] = 0.0
                        else:
                            add[year+'_'+quarter][df.loc[idx,'account_nm']] = float(df.loc[idx,'thstrm_amount'])                    
                else:
                    continue
        fins[corp] = add
    return fins

fins = get_fins_all()
