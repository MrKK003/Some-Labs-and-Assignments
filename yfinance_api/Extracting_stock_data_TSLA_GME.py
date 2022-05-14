import time
#from turtle import ht
import yfinance as yf
import pandas as pd
import requests
from bs4 import BeautifulSoup
import plotly.graph_objects as go
from plotly.subplots import make_subplots



def make_graph(stock_data, revenue_data, stock):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, subplot_titles=("Historical Share Price", "Historical Revenue"), vertical_spacing = .3)
    stock_data_specific = stock_data[stock_data.Date <= '2021--06-14']
    revenue_data_specific = revenue_data[revenue_data.Date <= '2021-04-30']
    fig.add_trace(go.Scatter(x=pd.to_datetime(stock_data_specific.Date, infer_datetime_format=True), y=stock_data_specific.Close.astype("float"), name="Share Price"), row=1, col=1)
    fig.add_trace(go.Scatter(x=pd.to_datetime(revenue_data_specific.Date, infer_datetime_format=True), y=revenue_data_specific.Revenue.astype("float"), name="Revenue"), row=2, col=1)
    fig.update_xaxes(title_text="Date", row=1, col=1)
    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_yaxes(title_text="Price ($US)", row=1, col=1)
    fig.update_yaxes(title_text="Revenue ($US Millions)", row=2, col=1)
    fig.update_layout(showlegend=False,
    height=900,
    title=stock,
    xaxis_rangeslider_visible=True)
    fig.show()
  
  
    
def extracting_tesla_stock_data():
    #Question 1
    tesla=yf.Ticker("TSLA")
    tesla_data=tesla.history(period='max')
    tesla_data.reset_index(inplace=True)
    print(tesla_data.head(5))
    
    #Question 2
    url="https://www.macrotrends.net/stocks/charts/TSLA/tesla/revenue"
    html_data=requests.get(url).text
    
    #print(html_data)
    #tesla_revenue=pd.read_html(html_data)
    #print(tesla_revenue)
    
    soup=BeautifulSoup(html_data,"html5lib")
    
    tesla_revenue=pd.DataFrame(columns=['Date','Revenue'])
    for row in soup.find_all('tbody')[1].find_all('tr'):
        col=row.find_all('td')
        date=col[0].text
        revenue=col[1].text
        
        tesla_revenue=tesla_revenue.append({'Date':date,'Revenue':revenue},ignore_index=True)
     
    #removing ,|\$    
    tesla_revenue["Revenue"] = tesla_revenue['Revenue'].str.replace(',|\$',"")
    
    #removing null or empty strings
    tesla_revenue.dropna(inplace=True)
    tesla_revenue = tesla_revenue[tesla_revenue['Revenue'] != ""]
    
    print(tesla_revenue.tail(5))
    return tesla_data,tesla_revenue
    
    
def extracting_GameStop_stock_data():
    #Question 3
    gamestop=yf.Ticker("GME")
    gme_data=gamestop.history(period="max")
    gme_data.reset_index(inplace=True)
    print(gme_data.head(5))
    
    #Question 4
    url="https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-PY0220EN-SkillsNetwork/labs/project/stock.html"
    html_data=requests.get(url).text
    
    soup=BeautifulSoup(html_data,"html5lib")
    
    gme_revenue=pd.DataFrame(columns=['Date','Revenue'])
    for row in soup.find_all('tbody')[1].find_all('tr'):
        col=row.find_all('td')
        date=col[0].text
        revenue=col[1].text
        
        gme_revenue=gme_revenue.append({'Date':date,'Revenue':revenue},ignore_index=True)
     
    #removing ,|\$    
    gme_revenue["Revenue"] = gme_revenue['Revenue'].str.replace(',|\$',"")
    
    #removing null or empty strings
    gme_revenue.dropna(inplace=True)
    gme_revenue = gme_revenue[gme_revenue['Revenue'] != ""]
    
    print(gme_revenue.tail(5))
    return gme_data,gme_revenue
    
       
    
def main():
    tesla_data,tesla_revenue=extracting_tesla_stock_data()
    
    #Question 5
    make_graph(tesla_data,tesla_revenue,'Tesla')
    
    gme_data,gme_revenue=extracting_GameStop_stock_data()
    
    #Question 6
    make_graph(gme_data,gme_revenue,'GameStop')
    

if __name__ == "__main__": 
    t1=time.perf_counter()
    main()
    t2=time.perf_counter()
    print(f'Finished in {t2-t1} seconds')