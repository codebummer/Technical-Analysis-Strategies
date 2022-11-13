import numpy as np
from scipy.stats import norm

r = 0.01
S = 30 # Underlying Price
K = 40 # Stock Price, or Discounted Strike Price
TIME = 240/365
sigma = 0.30

def blackScholes(r, S, K, TIME, sigma, type='CALL'):
    day1 = (np.log(S/K) + (r + sigma**2/2) * TIME) / (sigma * np.sqrt(TIME))
    day2 = day1 - sigma * np.sqrt(TIME)
    try:
        if type == 'CALL':
            price = S * norm.cdf(day1, 0, 1) - K * np.exp(-r*TIME) * norm.cdf(day2, 0, 1)
        elif type == 'PUT':
            price = K * np.exp(-r*TIME) * norm.cdf(-day2, 0, 1) - S * norm.cdf(-day1, 0, 1)
        return price
    except:
        print('Please confirm all option parameters above!!!')

print('Option Price is: ', round(blackScholes(r, S, K, TIME, sigma, type='CALL'), 2))
print('Option Price is: ', round(blackScholes(r, S, K, TIME, sigma, type='PUT'), 2))
