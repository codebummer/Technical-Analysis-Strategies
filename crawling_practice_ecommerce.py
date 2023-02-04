import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
import requests
from bs4 import BeautifulSoup
import time, warnings
from tqdm.notebook import tqdm
from datetime import datetime
import re
warnings.simplefilter('ignore')

query = '애플'
url = 'https://shopping.naver.com/home'
edge = webdriver.Edge()
edge.get(url)
time.sleep(2)
edge.find_element(By.XPATH, '//*[@id="__next"]/div/div[1]/div/div/div[2]/div/div[2]/div/div[2]/form/fieldset/div[1]/div/input').send_keys(query)
edge.find_element(By.XPATH, '//*[@id="__next"]/div/div[1]/div/div/div[2]/div/div[2]/div/div[2]/form/fieldset/div[1]/div/button[2]').click()
time.sleep(1)

results = {'제품':[], '가격':[], '리뷰':[], '별점':[], '구매건수':[], '등록일':[], '찜하기':[]}

for page in range(1,5):    
    edge.execute_script('window.scrollTo(0,document.body.scrollHeight)')    
    time.sleep(1)
    items = edge.find_elements(By.CLASS_NAME, 'basicList_inner__xCM3J')
    for item in items:        
        add = {'제품':'', '가격':0,'리뷰':0, '별점':0, '구매건수':0, '등록일':0, '찜하기':0}        
        frame = item.text

        add['제품'] = item.find_element(By.CLASS_NAME, 'basicList_title__VfX3c').text
        add['가격'] = int(item.find_element(By.CLASS_NAME, 'price_num__S2p_v').text.replace(',','').replace('원',''))

        footers = item.find_elements(By.CLASS_NAME, 'basicList_etc__LSkN_')

        for footer in footers:
            footnote = footer.text
            if '리뷰' in footnote:
                add['리뷰'] = int(footer.find_element(By.TAG_NAME, 'em').text.replace(',',''))
                if '별점' in footnote:
                    add['별점'] = footer.find_element(By.CLASS_NAME, 'basicList_star__UzKiv').text.split()[-1]
            elif '구매건수' in footnote:
                add['구매건수'] = int(footer.find_element(By.TAG_NAME, 'em').text.replace(',',''))
            elif '등록일' in footnote:
                add['등록일'] = datetime.strptime(footer.text.split()[-1],'%Y.%m.')
            elif '찜하기' in footnote:
                add['찜하기'] = int(footer.find_element(By.TAG_NAME, 'em').text.replace(',',''))

        for key in add.keys():
            results[key].append(add[key])

    edge.find_element(By.XPATH, f'//*[@id="__next"]/div/div[2]/div[2]/div[4]/div[1]/div[3]/div/a[{page}]').click()
    time.sleep(1)
        
results = pd.DataFrame(results)
results.iloc[:,1:]
