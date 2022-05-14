import time
from numpy import datetime_data
import plotly.graph_objects as go
import pandas as pd
from pycoingecko import CoinGeckoAPI
from plotly.offline import plot

cg = CoinGeckoAPI()

def main():
    bitcoin_data=cg.get_coin_market_chart_by_id(id='bitcoin', vs_currency='usd', days=30)
    bitcoin_price_data= bitcoin_data['prices']
    #print(bitcoin_price_data)
    data=pd.DataFrame(bitcoin_price_data,columns=['TimeStamp', 'Price'])
    #print(data)
    data['Date']=pd.to_datetime(data['TimeStamp'], unit='ms')
    #print(data)
    data=data[['Date','Price','TimeStamp']]
    del data['TimeStamp']
    #print(data)
    candlestick_data=data.groupby(data.Date.dt.date).agg({'Price':['min','max','first','last']})
    print(candlestick_data)
    fig=go.Figure(data=[go.Candlestick(x=candlestick_data.index, open=candlestick_data['Price']['first'],high=candlestick_data['Price']['max'],low=candlestick_data['Price']['min'],close=candlestick_data['Price']['last'])])
    fig.update_layout(xaxis_rangeslider_visible=False,xaxis_title='Data',yaxis_title='Price (USD $)',title='Bitcoin Candlestick Chart Over Past 30 Days')
    #fig.show()
    plot(fig,filename='bitcoin_candlestick_graph.html')
    


if __name__ == "__main__": 
    t1=time.perf_counter()
    main()
    t2=time.perf_counter()
    print(f'Finished in {t2-t1} seconds')
    