import time
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

def extracting_stock_data():
    #data=yf.download(tickers="AMD META NVDA TWTR AMZN GOOGL MSFT AAPL BLK SNAP ^GSPC ^DJI", period="5y", interval="1mo",group_by = 'ticker') #^GSPC AMZN TSLA BRK-A META TSM NVDA BAC AMD META NVDA TWTR AMZN GOOGL MSFT AAPL BLK SNAP ^GSPC ^DJI
    data=yf.download(tickers="USDCAD=X", period="1y", interval="1d",group_by = 'ticker')
    data.dropna(inplace=True)
    
    #close_data=data.iloc[:, data.columns.get_level_values(1)=='Close']
    #close_data.to_excel("stock_data_assignment_salamatin.xlsx")
    data.to_excel('stock_data_USDCAD.xlsx')
    
    
def main():
    extracting_stock_data()


if __name__ == "__main__": 
    t1=time.perf_counter()
    main()
    t2=time.perf_counter()
    print(f'Finished in {t2-t1} seconds')