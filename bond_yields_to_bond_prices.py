# convert bond yields to bonds prices
def calc_bond_price(assgined_amount, yields):
    '''
    aggsigned_amount: initial cash amount assigned to bonds in the working investment strategy,
    yields: a pandas series or dataframe that holds daily bond yields
    '''
    # calculate the inital fixed income amount from the initial cash amount assigned to invest in bonds
    # 
    # bonds yields amount (or fixed income amount) at the time of purchase: fixed_income (in the below code block)
    # = bond yield at the time of purchase (in decimal digits, not in percent) + 1(to include the principal) * the initial cash amount to invest in bonds
    fixed_income = assgined_amount * (yields.values[0]/100+1)

    # the later market prices of the above fixed income amount at the time of purchase
    # = [bond yield change (in decimal digits, not in percent)
    # * -1(to apply the negative relationship between bond prices and bond yields) 
    # + 1(to include the principal) ] * the initial fixed income amount at the time of purchase
    # Note: pandas.DataFrame.pct_change() does not generate changes in percent, but in decimal digits.
    bond_prices = yields.pct_change().multiply(-1).add(1) * fixed_income
    
    # fill the first row which got empty after pct_change() with the initial fixed income amount at the time of purchase
    bond_prices[0] = fixed_income
    return bond_prices
