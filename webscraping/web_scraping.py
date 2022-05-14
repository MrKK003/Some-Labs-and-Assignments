from bs4 import BeautifulSoup # this module helps in web scrapping.
import requests  # this module helps us to download a web page
import time
import pandas as pd

#Documentation
#https://www.crummy.com/software/BeautifulSoup/bs4/doc/?utm_medium=Exinfluencer&utm_source=Exinfluencer&utm_content=000026UJ&utm_term=10006555&utm_id=NA-SkillsNetwork-Channel-SkillsNetworkCoursesIBMDeveloperSkillsNetworkPY0220ENSkillsNetwork23455606-2021-01-01


def html_soup():
    html="<!DOCTYPE html><html><head><title>Page Title</title>\
        </head><body><h3><b id='boldest'>Lebron James</b></h3><p>\
            Salary: $ 92,000,000 </p><h3> Stephen Curry</h3><p>\
                Salary: $85,000, 000 </p><h3> Kevin Durant </h3><p>\
                    Salary: $73,200, 000</p></body></html>"
                    
    soup = BeautifulSoup(html, "html.parser")
    #print(soup.prettify())
    tag_object=soup.title
    tag_object=soup.h3
    #print("tag object:",tag_object)
    tag_child =tag_object.b
    #print(tag_child)
    
    parent_tag=tag_child.parent
    #print(parent_tag)
    #print(tag_object.parent)
    sibling_1=tag_object.next_sibling
    sibling_2=sibling_1.next_sibling
    #print(sibling_2.next_sibling)
    
    # If the tag has attributes, the tag <code>id="boldest"</code> has an attribute 
    # <code>id</code> whose value is <code>boldest</code>. You can access a tag’s attributes 
    # by treating the tag like a dictionary:
    id=tag_child['id']
    
    dic=tag_child.attrs
    #print(dic)
    
    #tag_child.get('id')
    
    tag_string=tag_child.string
    print(tag_string)

def html_table():
    table="<table><tr><td id='flight'>Flight No</td><td>Launch site</td>\
        <td>Payload mass</td></tr><tr> <td>1</td><td>\
            <a href='https://en.wikipedia.org/wiki/Florida'>Florida<a></td><td>300\
                kg</td></tr><tr><td>2</td><td><a href='https://en.wikipedia.org/wiki/Texas'>\
                    Texas</a></td><td>94 kg</td></tr><tr><td>3</td><td>\
                        <a href='https://en.wikipedia.org/wiki/Florida'>Florida<a> </td><td>80 kg</td>\
                            </tr></table>"
    table_bs = BeautifulSoup(table, "html.parser")
    table_rows=table_bs.find_all('tr') #list
    #print(table_rows)
    
    # for i,row in enumerate(table_rows):
    #     print("row",i)
    #     cells=row.find_all('td')
    #     for j,cell in enumerate(cells):
    #         print('colunm',j,"cell",cell)
    
    list_input=table_bs .find_all(name=["tr", "td"])
    #print(list_input)
    print(table_bs.find_all(href=True))
    
def web_scrapping_test():
    url = "http://www.ibm.com"
    data  = requests.get(url).text
    soup = BeautifulSoup(data,"html.parser")  # create a soup object using the variable 'data'
    
    # for link in soup.find_all('a',href=True):  # in html anchor/link is represented by the tag <a>
    #     print(link.get('href'))
    
    for link in soup.find_all('img'):# in html image is represented by the tag <img>
        print(link)
        print(link.get('src'))
    
def colors():
    url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-DA0321EN-SkillsNetwork/labs/datasets/HTMLColorCodes.html"
    # get the contents of the webpage in text format and store in a variable called data
    data  = requests.get(url).text
    soup = BeautifulSoup(data,"html.parser")
    #find a html table in the web page
    table = soup.find('table') # in html table is represented by the tag <table>
    #print(table.prettify)
    for row in table.find_all('tr'): # in html table row is represented by the tag <tr>
        # Get all columns in each row.
        cols = row.find_all('td') # in html a column is represented by the tag <td>
        color_name = cols[2].string # store the value in column 3 as color_name
        color_code = cols[3].string # store the value in column 4 as color_code
        print("{}--->{}".format(color_name,color_code))
        
def html_tables_to_dataframes():
    url = "https://en.wikipedia.org/wiki/World_population"
    # get the contents of the webpage in text format and store in a variable called data
    data  = requests.get(url).text
    soup = BeautifulSoup(data,"html.parser")
    #find all html tables in the web page
    tables = soup.find_all('table') # in html table is represented by the tag <table>
    #find specific string
    for index,table in enumerate(tables):
        if ("10 most densely populated countries" in str(table)):
            table_index = index
            #print(table_index)
    
    #print(tables[table_index].prettify())
    population_data = pd.DataFrame(columns=["Rank", "Country", "Population", "Area", "Density"])

    for row in tables[table_index].tbody.find_all("tr"):
        col = row.find_all("td")
        if (col != []):
            rank = col[0].text
            country = col[1].text.strip()
            population = col[2].text.strip()
            area = col[3].text.strip()
            density = col[4].text.strip()
            population_data = population_data.append({"Rank":rank, "Country":country, "Population":population, "Area":area, "Density":density}, ignore_index=True)

    #print(population_data)
    
    population_data_read_html = pd.read_html(str(tables[5]), flavor='bs4')[0]
    print(population_data_read_html)
    
    dataframe_list = pd.read_html(url, flavor='bs4') #alternative to findd_all
    print(dataframe_list)
    #pd.read_html(url, match="10 most densely populated countries", flavor='bs4')[0] #alternative to 102-105
    
    


def main():
    #html_soup()
    #html_table()
    #web_scrapping_test()
    #colors()
    html_tables_to_dataframes()

    

    

    


if __name__ == "__main__": 
    t1=time.perf_counter()
    main()
    t2=time.perf_counter()
    print(f'Finished in {t2-t1} seconds')