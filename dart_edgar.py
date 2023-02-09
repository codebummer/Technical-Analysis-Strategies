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
# import requests, json, xmltodict
from  urllib.request import urlopen
from selenium import webdriver
sys.path.append('\myprojects\MarketDB')
from cert_key import *
sns.set_theme(style='ticks')

# DART link
url = 'https://opendart.fss.or.kr/api/corpCode.xml?crtfc_key='
param = CERT_KEY

os.chdir(r'D:\myprojects')
# Run when you don't have the entire company list
# This list also contains unlisted companies 
with urlopen(url+param) as response:
    with zipfile.ZipFile(io.BytesIO(response.read())) as zip:
        zip.extractall('./MarketDB/corpCode')
        print('All Business List Successfully Received')

# Read the existing company list
corps = pd.read_xml('./MarketDB/corpCode/CORPCODE.xml')
corps.dropna(inplace=True)
corps = corps.reset_index().drop('index', axis='columns')
# remove = corps.index[corps['corp_name']=='루트원플러스']
corps = corps.drop(remove)

dart = OpenDartReader(CERT_KEY)
# reports = {'1분기보고서':'11013', '반기보고서':'11012', '3분기보고서':'11014', '사업보고서':'11011'}
reports = {'3분기보고서':'11014'}
for corp in corps['corp_name']:
# for corp in corps['corp_name'].values[remove[0]:]:
    # for year in range(2022, 2014, -1):
    year = 2022
    for title, report in reports.items():
        try:
            add = dart.finstate_all(corp, year, report)
            time.sleep(0.5)
        except:
            time.sleep(0.5)
            continue
        if len(add) != 0:
            with sqlite3.connect('./MarketDB/fins.db') as db:
                add.to_sql(corp+'_'+title, db, if_exists='replace', index=False)
            # dart.finstate_all(code, 2021, report, 'OFS')

with sqlite3.connect('./MarketDB/fins.db') as db:
    query = '''SELECT * FROM sqlite_master WHERE type='table';'''
    tables = db.cursor().execute(query).fetchall()

# collect all listed company names that have financial data
listed_corps = [table[1].split('_')[0] for table in tables]

# EDGAR 
edge = webdriver.Edge()
edge.get('https://www.sec.gov/Archives/edgar/daily-index/xbrl/companyfacts.zip')
edge.close()

with zipfile.ZipFile(r'C:\Users\ACECOM\Downloads\companyfacts.zip') as zip:
    zip.extractall(r'D:\myprojects\MarketDB\companyfacts')
os.remove(r'C:\Users\ACECOM\Downloads\companyfacts.zip')

with open(r'D:\myprojects\MarketDB\companyfacts\CIK0000001750.json') as opened:
    file = json.load(opened)
file['entityName']
