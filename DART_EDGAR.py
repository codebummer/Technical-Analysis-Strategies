import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os, csv
from tqdm import tqdm
import OpenDartReader
import zipfile, io, sys, json
import sqlite3
import time
from tqdm import tqdm
# import requests, json, xmltodict
from  urllib.request import urlopen
from selenium import webdriver
sys.path.append('\myprojects\MarketDB')
from DART_key import *
sns.set_theme(style='ticks')

# DART link
url = 'https://opendart.fss.or.kr/api/corpCode.xml?crtfc_key='
param = DART_KEY

os.chdir(r'D:\myprojects')

# Read the existing company list
def get_company_list():
    # Run when you don't have the entire company list
    # This list also contains unlisted companies 
    with urlopen(url+param) as response:
        with zipfile.ZipFile(io.BytesIO(response.read())) as zip:
            zip.extractall('./MarketDB/corpCode')
            print('All Business List Successfully Received')
    corps = pd.read_xml('./MarketDB/corpCode/CORPCODE.xml')
    corps.dropna(inplace=True)
    corps = corps.reset_index().drop('index', axis='columns')
    return corps
    # remove = corps.index[corps['corp_name']=='루트원플러스']
    # corps = corps.drop(remove)

# Read the prior database downloaded 
with sqlite3.connect('./MarketDB/KoreaFins/fins_2022_3분기보고서.db') as db:
    query = '''SELECT * FROM sqlite_master WHERE type='table';'''
    tables = db.cursor().execute(query).fetchall()

# collect all listed company names that have financial data
listed_corps = [table[1].split('_')[0] for table in tables]

# instantiate the Open DART class
dart = OpenDartReader(DART_KEY)

years = [2021, 2020, 2019, 2018]

# reports = {'1분기보고서':'11013', '반기보고서':'11012', '3분기보고서':'11014', '사업보고서':'11011'}
reports = {'사업보고서':'11011', '3분기보고서':'11014', '반기보고서':'11012', '1분기보고서':'11013'}
filenames = {'사업보고서':'Q4', '3분기보고서':'Q3', '반기보고서':'Q2', '1분기보고서':'Q1'}

for year in tqdm(years):
    for title, report in tqdm(reports.items()):
    # for corp in corps['corp_name']:
    # for corp in corps['corp_name'].values[remove[0]:]:
        # for year in range(2022, 2014, -1):
        for corp in tqdm(listed_corps):
            try:
                add = dart.finstate_all(corp, year, report)
                # without time.sleep(0.5), an error will occur
                time.sleep(0.5)
            except:
                time.sleep(0.5)
                continue
            if len(add) != 0:
                with sqlite3.connect(f'./MarketDB/{year}_{filenames[title]}.db') as db:
                    add.to_sql(corp+'_'+title, db, if_exists='replace', index=False)
                # dart.finstate_all(code, 2021, report, 'OFS')


stocks_info = {}
fields = ['주식총수', '배당', '조건부자본증권미상환', '회사채미상환', '단기사채미상환', '미등기임원보수']
for corp in tqdm(listed_corps):
    for field in fields:
        if corp not in stocks_info.keys():
            stocks_info[corp] = [dart.report(corp, field, 2022)]
        else:
            stocks_info[corp].append(dart.report(corp, field, 2022))
        # without time.sleep(0.5), an error will occur
        time.sleep(0.5)

# EDGAR 
edge = webdriver.Edge()
edge.get('https://www.sec.gov/Archives/edgar/daily-index/xbrl/companyfacts.zip')
edge.close()

with zipfile.ZipFile(r'C:\Users\ACECOM\Downloads\companyfacts.zip') as zip:
    zip.extractall(r'D:\myprojects\MarketDB\companyfacts')
os.remove(r'C:\Users\ACECOM\Downloads\companyfacts.zip')

# make a list of all listed companies
nyse = {}
files = os.listdir(r'D:\myprojects\MarketDB\companyfacts')
for file in tqdm(files[14618:]):
    with open(r'D:\myprojects\MarketDB\companyfacts\\'+file, encoding='utf-8') as opened:
        company = json.load(opened)
        # cik = '0'*(10-len(str(company['cik']))) + str(company['cik'])
        try:
            nyse[company['cik']] = company['entityName']       
        except:
            continue

with open(r'D:\myprojects\MarketDB\nyse_list.json', 'w') as file:
    json.dump(nyse, file)
