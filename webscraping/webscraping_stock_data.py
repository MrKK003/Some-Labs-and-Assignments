import time
from matplotlib.pyplot import text
import pandas as pd
import requests
from bs4 import BeautifulSoup





def extracting_netflix_stock_data():
    url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-PY0220EN-SkillsNetwork/labs/project/netflix_data_webpage.html"
    data  = requests.get(url).text 
    
    soup = BeautifulSoup(data, 'html5lib') #parse into html
    
    netflix_data = pd.DataFrame(columns=["Date", "Open", "High", "Low", "Close", "Volume"])
    
    # First we isolate the body of the table which contains all the information
    # Then we loop through each row and find all the column values for each row
    for row in soup.find("tbody").find_all('tr'):
        col = row.find_all("td")
        date = col[0].text
        Open = col[1].text
        high = col[2].text
        low = col[3].text
        close = col[4].text
        adj_close = col[5].text
        volume = col[6].text
        
        # Finally we append the data of each row to the table
        netflix_data = netflix_data.append({"Date":date, "Open":Open, "High":high, "Low":low, "Close":close, "Adj Close":adj_close, "Volume":volume}, ignore_index=True)    
    
    print(netflix_data)
    

def extracting_amazon_stock_data():
    url='https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-PY0220EN-SkillsNetwork/labs/project/amazon_data_webpage.html'
    html_data=requests.get(url).text
    
    soup=BeautifulSoup(html_data,'html5lib')
    
    #Question 1
    tag_object=soup.title.text
    print(tag_object)
    
    amazon_data = pd.DataFrame(columns=["Date", "Open", "High", "Low", "Close", "Volume"])

    for row in soup.find("tbody").find_all("tr"):
        col = row.find_all("td")
        date = col[0].text
        Open = col[1].text
        high = col[2].text
        low = col[3].text
        close = col[4].text
        adj_close = col[5].text
        volume = col[6].text
        
        amazon_data = amazon_data.append({"Date":date, "Open":Open, "High":high, "Low":low, "Close":close, "Adj Close":adj_close, "Volume":volume}, ignore_index=True)
    
    print(amazon_data.head(5))
    
    #Question 2
    print(amazon_data.columns)
    
    #Question 3
    #print(amazon_data.info())
    #print(amazon_data['Date'])

    amazon_data1=amazon_data.set_index('Date')
    print(amazon_data1)
    print(amazon_data1.loc['Jun 01, 2019','Open'])
    


def main():
    extracting_netflix_stock_data()
    #extracting_amazon_stock_data()
    

if __name__ == "__main__": 
    t1=time.perf_counter()
    main()
    t2=time.perf_counter()
    print(f'Finished in {t2-t1} seconds')