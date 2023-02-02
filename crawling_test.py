import requests
import bs4
import pandas as pd
from datetime import datetime
import numpy as np
from selenium import webdriver
from selenium.webdriver.common.by import By
import time

page_no = 2
url = f'https://finance.naver.com/sise/sise_index_day.naver?code=KPI200&page={page_no}'

source = requests.get(url).text
source = bs4.BeautifulSoup(source) # generating a parse tree
print(source.prettify())
prices = source.find_all('td', class_='number_1')
dates = source.find_all('td', class_='date')
dates[0].text

date_list = []
for date in dates:
    date_list.append(date.text)
date_list

price_list = []
for price in prices[::4]:
    price_list.append(price.text)
price_list

last_url = source.find_all('td', class_='pgRR')[0].find_all('a')[0]['href']
last_page = int(last_url.split('&page=')[-1])

date_list = []
price_list = []
for page in range(1, last_page+1):
    url = f'https://finance.naver.com/sise/sise_index_day.naver?code=KPI200&page={page}'
    source = requests.get(url).text
    source = bs4.BeautifulSoup(source) # generating a parse tree
    prices = source.find_all('td', class_='number_1')
    dates = source.find_all('td', class_='date')

    for date in dates:
        try:
            date_list.append(datetime.strptime(date.text, '%Y.%m.%d').strftime('%Y-%m-%d'))
        except:
            date_list.append(date.text)

    for price in prices[::4]:
        try:
            price_list.append(float(price.text))
        except:
            price_list.append(price.text)

df = pd.DataFrame({'date':date_list, 'price':price_list}).dropna()
df.to_excel('kospi200.xlsx', index=False)



# Dynamic Page Crawling
from selenium import webdriver
from selenium.webdriver.common.by import By
import time

driver = webdriver.Chrome('chromedriver')

# go to opinet.co.kr
driver.get('https://www.opinet.co.kr/searRgSelect.do')
time.sleep(2)

# go to 싼 주유소 찾기 page
driver.execute_script('goSubPage(0,0,99)')
time.sleep(2)

# get the province list
metros_elem = driver.find_element(By.XPATH, '//*[@id="SIDO_NM0"]')
metros_tags = metros_elem.find_elements(By.TAG_NAME, 'option')
metros = []
for tag in metros_tags:
    metros.append(tag.get_attribute('value'))
metros = metros[1:]

def get_cities(driver):
    cities_elem = driver.find_element(By.XPATH, '//*[@id="SIGUNGU_NM0"]')
    cities_tags = cities_elem.find_elements(By.TAG_NAME, 'option')
    cities = []
    for tag in cities_tags:
        cities.append(tag.get_attribute('value'))
    return cities[1:]

for metro in metros:
    # get the metro element every time you repeat, or it will cause an error (not recognize the element reference)
    # send an input to an element identified by Xpath
    driver.find_element(By.XPATH, '//*[@id="SIDO_NM0"]').send_keys(metro)
    time.sleep(2)
    cities = get_cities(driver)
    for city in cities:
        driver.find_element(By.XPATH, '//*[@id="SIGUNGU_NM0"]').send_keys(city)
        # search button click
        driver.find_element(By.XPATH, '//*[@id="searRgSelect"]').click()
        time.sleep(2)
        # excel button click
        driver.find_element(By.XPATH, '//*[@id="glopopd_excel"]').click()    
        time.sleep(3)        


# Edge version
edge = webdriver.Edge()
time.sleep(2)
edge.get('https://www.opinet.co.kr/searRgSelect.do')
time.sleep(5)
edge.execute_script('goSubPage(0,0,99)')
time.sleep(2)

metros_elem = edge.find_element(By.XPATH, '//*[@id="SIDO_NM0"]')
metros_tags = metros_elem.find_elements(By.TAG_NAME, 'option')
metros = []
for tag in metros_tags:
    metros.append(tag.get_attribute('value'))
metros = metros[1:]

def get_cities(edge):
    cities_elem = edge.find_element(By.XPATH, '//*[@id="SIGUNGU_NM0"]')
    cities_tags = cities_elem.find_elements(By.TAG_NAME, 'option')
    cities = []
    for tag in cities_tags:
        cities.append(tag.get_attribute('value'))
    return cities[1:]    

for metro in metros:
    # get the metro element every time you repeat, or it will cause an error (not recognize the element reference)
    # send an input to an element identified by Xpath
    edge.find_element(By.XPATH, '//*[@id="SIDO_NM0"]').send_keys(metro)
    time.sleep(2)
    cities = get_cities(edge)
    for city in cities:
        edge.find_element(By.XPATH, '//*[@id="SIGUNGU_NM0"]').send_keys(city)
        # search button click
        edge.find_element(By.XPATH, '//*[@id="searRgSelect"]').click()
        time.sleep(2)
        # excel button click
        edge.find_element(By.XPATH, '//*[@id="glopopd_excel"]').click()    
        time.sleep(3)       
