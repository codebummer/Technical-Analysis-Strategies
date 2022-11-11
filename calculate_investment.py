import pandas as pd
from datetime import datetime
import re
from math import floor

class InvestEval():
    def __init__(self, past_orders=pd.DataFrame([])):
        '''
        In case there are past trasanction records and you want to use and continue trading with them,
        pass on the past information in the form of an object when instantiating the object. 
        Required information is as follows:
        
        past_orders = an object pointer for the dataframe that holds the entire past records
          
        past_buyID = a unique counter or ID numbers recorded to identify buy orders held in the above past_orders dataframe.
 
        past_sellID = the same as above
        '''               
        self.orders = past_orders
        self.size_of_order = [0, 0, 0]
        #self.size_of_order is a list that keeps 3 elements to track the amount or the number of shares for each order.
        #self.size_of_order[0]: the amount of each order floored by the mutiplication of price and shares
        #self.size_of_order[1]: the number of shares for each order. This will be the primary indicator to use in trading.
        #self.size_of_order[2]: the price of one share for each order
        
        
        #The following if blocks determine whether there are any past records passed on and process accordingly.
        
        #When there are no past records passed on
        if self.orders.empty:
            #self.invested amount: totoal amount of investment so far
            #self.current_cash: total cash amount not invested yet
            #self.commission: total amount of fees and taxes for making transactions and profits thereof
            self.invested_amount, self.current_cash = 0, 0
            self.commission = 0.0014
            
            #Decided to use separate counter or identifier variables for buy orders and sell orders
            self.buyID = self.sellID = 0
        
        #When there are past records,  
        else:
            #But, when there are only 'BUY' orders made
            if 'OrderID' in self.orders.columns: #if 'OrderID' in self.order: also works    
                self.invested_amount = self.orders['InvestAmount'].values[-1]
                self.current_cash = self.orders['CurrentCash'].values[-1]
                self.commission = 0.0014
                
                # buy_rows = self.orders[self.orders['OrderType']=='BUY']
                # last_buy = buy_rows['OrderID'].values[-1]
                # self.buyID = last_buy
                # Those above three lines are equivalent to the below one line
                self.buyID = self.orders[self.orders['OrderType']=='BUY']['OrderID'].values[-1]   
                
                #When there are also 'SELL' orders made as well
                if 'SELL' in self.orders.values:
                    self.sellID = self.orders[self.orders['OrderType']=='SELL']['OrderID'].values[-1]

                
                #When there are no 'SELL' orders made. This block works with the above only 'BUY' block.
                elif 'SELL' not in self.orders.values:
                    pass
                
                    # The following part of code is not recommended to use. 
                    # However those are kept just to remind how my thought process flew
                    
                    # last_order_row = self.orders.loc[len(self.orders['OrderID'])-1]
                    # if last_order_row['OrderType'] == 'BUY':
                    #     self.buyID = last_buy
                    # self.buyID = self.orders['OrderID'].values[-1]                   
                    # self.sellID = self.orders['OrderID'].values[-1]  
                    

            #When there are no past records of orders made, but only investment having been set.
            if 'InvestAmount' in self.orders.columns: 
                # Do not use 'elif' here, because 'elif' here will make this 'elif or if' block part of the above 'if' block.
                # That makes pass' exit the entire 'if~elif' block including this one, when the immediately above 'elif' evaluates to True. 
                # To avoid exiting the entire block including this one by the above 'pass',
                # this 'if' block should be a separate block from the above big block of 'if'.
                self.invested_amount = self.orders['InvestAmount'].values[-1]
                self.current_cash = self.orders['CurrentCash'].values[-1]                   
                self.buyID, self.sellID = 0, 0
                
                self.commission = 0.0014
             

    def get_orders(self):
        return self.orders       
    
    def setcash(self, cash):
        self.invested_amount += cash
        self.current_cash += cash
        self.orders['InvestAmount'] = [self.invested_amount]
        self.orders['CurrentCash'] = [self.current_cash]

    def _log_current_cash(self, order, price, shares):
        if order == 'BUY':
            self.current_cash -= price * shares
            self.orders['CurrentCash'] = [self.current_cash]
        elif order == 'SELL':
            self.current_cash += price * shares
            self.orders['CurrentCash'] = [self.current_cash]

        self.orders['InvestAmount'] = [self.invested_amount]

        # if order == 'CASHIN':
        #     self.orders['CurrentCash'] = cash
        # elif order == 'BUY':
        #     self.current_cash -= self.orders['Price'].values[-1] * self.orders['Shares'].values[-1]
        #     self.orders['CurrentCash'] = self.current_cash
        # elif order == 'SELL':
        #     self.current_cash += self.orders['Price'].values[-1] * self.orders['Shares'].values[-1]
        #     self.orders['CurrentCash'] = self.current_cash
        '''
        When a new value is added to a dataframe, it can be indexed like 
        orders['Price'] = 1000
        However, when a currently existing value, especially the most recent one, is indexed, it should be located like 
        order['Price'].values[-1]
        because the first form gives the entire column when you need one value from that column.
        When you use the second form, it will give you a Numpy array 
        which can be individually indexed with parentheses().
        '''
    
    def setcomm(self, comm=0.0014):
        self.commission = comm
    
    def get_size_of_order(self):        
        return self.size_of_order    
    
    def set_size_of_order(self, size_of_order):
        self._calculate_size_of_order(size_of_order)
    
    def _calculate_size_of_order(self, size_of_order=[], price=0): 
        # 'price=0' is after 'size_of_order=0', because 'set_size_of_order' only gives one value while the current function takes only one.
        # Except when a user manually input the arguments, normally 'set_size_of_order' will take the 'price' value sent,
        # unless the order of the arguments are 'size_of_order=0' first and 'price=0' later.      
       
        #The argument 'price' is still in the function even though the primary calling function 'set_size_of_order' does not give that value.
        #This is to keep the opportunity open that this function can be used directly and manually in the Python interpreter.
        #In case it is not directly used by the user and the 'price' as an argument is not used,
        #This function will use 'self.size_of_order[2]' value for the price of one share for each order.
        
        #When the money amount is input for each order
        
        # if 'W' in size_of_order[1] or 'w' in size_of_order[1]: -> this block is equivalent to the below one block of an if.
        #     self.size_of_order[0] = int(re.sub('[a-zA-Z]', '', size_of_order))
        #     self.size_of_order[1] = self._calculate_shares_of_order(price, self.size_of_order[0])
        #     print(f'The amount of each order is set at {self.size_of_order[0]}KRW.'
        #           f'It is {self.size_of_order[0]/self.current_cash*100}% of the total current cash, {self.current_cash}KRW.\n'
        #           f'It can order {self.size_of_order[1]} share(s).')
        
        #In case the size_of_order is input directly from a user
        if len(size_of_order) == 1:
            size_of_order = self._number_character_splitter(size_of_order)
            
        if re.findall('[Ww]', size_of_order[1]):
            self.size_of_order[0] = size_of_order[0]
            self.size_of_order[1] = self._calculate_shares_of_order(price, self.size_of_order[0])
            print(f'The amount of each order is set at {self.size_of_order[0]}KRW.'
                  f'It is {self.size_of_order[0]/self.current_cash*100}% of the total current cash, {self.current_cash}KRW.\n'
                  f'It can order {self.size_of_order[1]} share(s).')
            
        #When the percentage amount is input for each order
        elif '%' in size_of_order[1]:
            self.size_of_order[0] = size_of_order[0] / 100 * self.current_cash
            self.size_of_order[1] = self._calculate_shares_of_order(price, self.size_of_order[0])
            print(f'The amount of each order is set at {self.size_of_order[0]}KRW.'
                  f'It is {size_of_order[0]}% of the total current cash, {self.current_cash}KRW.\n'
                  f'It can order {self.size_of_order[1]} share(s).')
        
        #When the number of shares is input for each order
        elif re.findall('[Ss]', size_of_order[1]):
        # elif 'S' in [size_of_order] or 's' in [size_of_order]:        
            # if price:
            if self.size_of_order[2]: #If the price of one share for each order has been input
                # self.size_of_order[0] = int(re.sub('[a-zA-Z', '', size_of_order)) * self.orders[self.orders['OrderID']==self.buyID]['Price']
                self.size_of_order[0] = size_of_order[0] * self.size_of_order[2] # In case this doesn't work. Use int(self.size_of_order[2])
                self.size_of_order[1] = size_of_order[0]
                print(f'The amount of each order is set at {self.size_of_order[0]}KRW.'
                  f'It is {self.size_of_order[0]/self.current_cash*100}% of the total current cash, {self.current_cash}KRW.\n'
                  f'It can order {size_of_order[0]} share(s).')
            
            #When the number of shares is input but the price is not input
            else:
                print(f'The price of a share has not been input. Please set the price first.')
                return
        
        #When the input is not made in the forms of 'W' or '%' or 'S'
        else:
            print(f'Input Value Error. Please input the amount in the forms of "10000W" or "10000w" or "10%" or "50s"')
            return    

    def make_order(self, order, price=0, size_of_order=None):
        if size_of_order == None:
            print('The amount of each order has not been set. Please input the WON amount or percentage amount.\nPlease input the amount in the forms of "10000W" or "10000w" or "10%"')
            return
        elif price == 0:
            print('The price of one share for each order has not been set. Please input the WON amount.')
            return
        
        if order == 'BUY':
            self.size_of_order[2] = price
            size_of_order = self._number_character_splitter(size_of_order)
            self.set_size_of_order(size_of_order)
            shares = self._calculate_shares_of_order(price, self.size_of_order[0])
            #Reject the order if the current cash is less than 5%  after making an order 
            if self.current_cash < price*shares*1.05: 
                #1.05 is multiplied for paying fees and taxes
                
                print('Cash is insufficient to make the order')
                return 
                #you can use the return statement without any parameter to exit a function        

            self._log_orders(order, price, shares, datetime.today())
            self._log_current_cash(order, price, shares)

        elif order == 'SELL':
            # self.size_of_order[2] = price            
            self._log_orders(order, price, shares, datetime.today())
            self._log_current_cash(order, price, shares)
            
            self._calculate_profit(self.orders['OrderID'].values[-1])
    
    def _number_character_splitter(self, size_of_order):
        return int(re.findall('^[0-9]+', size_of_order)[0]), re.findall('[a-zA-Z]+|%', size_of_order)[0]  
    
    def _calculate_shares_of_order(self, price, size_of_order):
        #_calculate_shares_of_order takes size_of_order, a local variable, instead of self.size_of_order, a global variable.
        #That's because it might be useful to use this method directly from the Python interpreter line,
        #taking a direct input value from it, even though it is named with '_' before it.
        return floor(size_of_order/price)

    def _log_orders(self, order, price, shares, time):       
        self._order_tracker(order)
        if order == 'BUY':
            self.orders['OrderType'] = ['BUY']
        if order == 'SELL':
            self.orders['OrderType'] = ['SELL']
        self.orders['Time', 'Price', 'Shares'] = [time, price, shares]     
     
    def _order_tracker(self, order):
        if order == 'BUY':
            self.buyID += 1
            self.orders['OrderID'] = [self.buyID]

        if order == 'SELL':            
            self.sellID += 1
            self.orders['OrderID'] = [self.sellID]
             
    def _calculate_profit(self, order_ID):
        rows = self._find_paired_order(order_ID)
        if len(rows) == 0:
            print('No order is made yet. Profits cannot be calculated.')
            return
        elif len(rows) == 1:
            print('A sell order is not made for the matching buy order yet. Profits cannot be calculated.')
            return
        
        buyrow, sellrow = rows[0], rows[1]
        profit_loss = self.orders.loc[sellrow, 'Price'] * self.orders.loc[sellrow, 'Shares'] / self.orders.loc[buyrow, 'Price'] * self.orders.loc[buyrow, 'Shares'] - 1
        self.orders.loc[[buyrow, sellrow], 'PL'] = [profit_loss]

    def _find_paired_order(self, order_ID):
        return self.orders[self.orders['OrderID'] == order_ID].index
        # rowID = self.orders['OrderID'].str.findall(order_ID) : This was the second choice of the above implementation  

invested = InvestEval()
invested.get_orders()

a = {'InvestAmount': [1000, 2000, 3000], 'CurrentCash': [100, 100, 300]}
test = pd.DataFrame(a)
'InvestAmount' in test
'InvestAmount' in test.columns
test[test['CurrentCash']==100]['CurrentCash'].values
re_invest = InvestEval(test)
re_invest.get_orders()
re_invest.make_order('BUY', '10000', '50s')


#The following snippet of code is to understand and practice pandas dataframes by playing with them
#It has nothing to do with the main code itself. 
a = {'InvestAmount': [1000, 2000, 3000], 'CurrentCash': [100, 100, 300]}
test = pd.DataFrame(a)
for key, value in test.items():
    print(key, value)


'InvestAmount' in test
'InvestAmount' in test.columns
test[test['CurrentCash']==100]['CurrentCash'].values


 
if True:
    print('a')
    if True:
        print('b')
        if False:
            print('c')
        elif True:
            pass
    if True:
        print('d')
print('e')

s = '5000000s'
t = '50%'
re.findall('[a-zA-Z]+|%', s)[0]
