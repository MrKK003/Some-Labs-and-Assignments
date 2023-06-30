import time
import string
import pandas as pd
import numpy as np
import pandas_datareader as pdr
import seaborn as sns
import matplotlib
from matplotlib import pyplot as plt

    
def main():
    path='/Users/kirillkiptyk/Downloads/ux-marketresults-3.csv'
    data=pd.read_csv(path,header=None)
    #print(df)
    
    data.columns=[num+1 for num in range(21)]
    #print(df.dtypes)
    df=data[[1,2,8]]
    df.columns=['Date','Code','Price']

    df.dropna(subset=['Price'], axis=0, inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    df['Date']=pd.to_datetime(df['Date'], dayfirst=True)
    
    df=df.sort_values(by='Date')
    df.reset_index(drop=True, inplace=True)
    
    df['day'] = df['Date'].dt.day
    df['month'] = df['Date'].dt.month
    df['year'] = df['Date'].dt.year
    
    df.drop('Date', inplace=True, axis=1)
    
    df3=df.groupby(['Code','year','month'])['Price'].mean()

    df3.to_csv("ukrmarketresults2")

    
    
    
if __name__ == "__main__": 
    t1=time.perf_counter()
    main()
    t2=time.perf_counter()
    print(f'Finished in {t2-t1} seconds')