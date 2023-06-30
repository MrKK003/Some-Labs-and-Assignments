import time
import numpy as np
import pandas as pd
import string


def main():
    path='/Users/kirillkiptyk/Downloads/ux-marketresults.csv'
    data=pd.read_csv(path,header=None)
    #print(df)
    
    data.columns=[num+1 for num in range(15)]
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
    
    df3=df.groupby(['year','month','Code'])['Price'].mean()
    
    #print(df3.head(50))
    print(df3)
    #print(df2['Code'].unique())
    #print(df['Code'].mode())
    
    

    
    
    
if __name__ == "__main__": 
    t1=time.perf_counter()
    main()
    t2=time.perf_counter()
    print(f'Finished in {t2-t1} seconds')