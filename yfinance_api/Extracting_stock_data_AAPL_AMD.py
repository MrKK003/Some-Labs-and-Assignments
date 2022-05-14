#from plotly.offline import plot
import time
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt


def extracting_apple_stock_data():
    apple = yf.Ticker("AAPL")
    apple_info=apple.info
    #print(apple_info)
    
    
    apple_share_price_data = apple.history(period="max") #pandas dataframe
    apple_share_price_data.reset_index(inplace=True)
    #print(apple_share_price_data)
    
    apple_share_price_data.plot(x="Date", y="Open")
    plt.show()

    
def extracting_amd_stock_data():    
    amd = yf.Ticker("AMD")
    amd_info=amd.info #dict of all info
    
    #Question 1
    country=amd_info['country']
    print(country)
    
    #Question 2
    sector=amd_info['sector']
    print(sector)
    
    #Question 3
    amd_share_price_data=amd.history(period="max")
    amd_share_price_data.reset_index(inplace=True)
    #print(amd_share_price_data)
    print(amd_share_price_data.loc['Jun 01, 2019','Volume'])
    
    

def main():
    #extracting_apple_stock_data()
    extracting_amd_stock_data()
    
    
    




if __name__ == "__main__": 
    t1=time.perf_counter()
    main()
    t2=time.perf_counter()
    print(f'Finished in {t2-t1} seconds')