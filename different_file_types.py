from PIL import Image 
from IPython.display import display
import time
import pandas as pd
import numpy as np
import json
import urllib.request
import xml.etree.ElementTree as ET
import xml.etree.ElementTree as etree
import matplotlib.pyplot as plt



#https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html?utm_medium=Exinfluencer&utm_source=Exinfluencer&utm_content=000026UJ&utm_term=10006555&utm_id=NA-SkillsNetwork-Channel-SkillsNetworkCoursesIBMDeveloperSkillsNetworkPY0101ENSkillsNetwork19487395-2022-01-01    
    
    
def csv_file_reading():
    url ='https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-PY0101EN-SkillsNetwork/labs/Module%205/data/addresses.csv'
    df = pd.read_csv(url,header=None)
    df.columns =['First Name', 'Last Name', 'Location ', 'City','State','Area Code']
    #print(df)
    df1=pd.DataFrame(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), columns=['a', 'b', 'c'])
    df1 = df1.transform(func = lambda x : x + 10)
    print(df1)
    result = df1.transform(func = ['sqrt'])
    print(result)


def json_file_reading_writing():
    person = {
        'first_name' : 'Mark',
        'last_name' : 'abc',
        'age' : 27,
        'address': {
            "streetAddress": "21 2nd Street",
            "city": "New York",
            "state": "NY",
            "postalCode": "10021-3100"
        }
    }
    
    with open('person.json', 'w') as f:  # writing JSON object
        json.dump(person, f)
    
    json_object = json.dumps(person, indent = 4) 
    
    # Writing to sample.json 
    with open("sample.json", "w") as outfile: 
        outfile.write(json_object) 
    
    #print(json_object)
    
    # Opening JSON file 
    with open('sample.json', 'r') as openfile: 
        # Reading from json file 
        json_object = json.load(openfile) 
  
    print(json_object) 
    print(type(json_object)) 
    

def xlsx_file_reading():
    urllib.request.urlretrieve("https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-PY0101EN-SkillsNetwork/labs/Module%205/data/file_example_XLSX_10.xlsx", "sample.xlsx")
    df = pd.read_excel("sample.xlsx")
    print(df)
    
    
def xml_file_reading():
    # create the file structure
    employee = ET.Element('employee')
    details = ET.SubElement(employee, 'details')
    first = ET.SubElement(details, 'firstname')
    second = ET.SubElement(details, 'lastname')
    third = ET.SubElement(details, 'age')
    first.text = 'Shiv'
    second.text = 'Mishra'
    third.text = '23'

    # create a new XML file with the results
    mydata1 = ET.ElementTree(employee)
    # myfile = open("items2.xml", "wb")
    # myfile.write(mydata)
    with open("new_sample.xml", "wb") as files:
        mydata1.write(files)
    
    tree = etree.parse("Sample-employee-XML-file.xml")

    root = tree.getroot()

    columns = ["firstname", "lastname", "title", "division", "building","room"]

    datatframe = pd.DataFrame(columns = columns)

    for node in root: 

        firstname = node.find("firstname").text

        lastname = node.find("lastname").text 

        title = node.find("title").text 
        
        division = node.find("division").text 
        
        building = node.find("building").text
        
        room = node.find("room").text
        
        datatframe = datatframe.append(pd.Series([firstname, lastname, title, division, building, room], index = columns), ignore_index = True)
    
    print(datatframe)
       
       
def image_file_reading():
    urllib.request.urlretrieve("https://hips.hearstapps.com/hmg-prod.s3.amazonaws.com/images/dog-puppy-on-garden-royalty-free-image-1586966191.jpg", "dog.jpg")
    
    # Read image 
    img = Image.open('dog.jpg') 
    
    # Output Images 
    display(img)
    
    
def dataset_analysis():
    path = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-PY0101EN-SkillsNetwork/labs/Module%205/data/diabetes.csv"
    df = pd.read_csv(path) 
    
    # show the first 5 rows using dataframe.head() method
    print("The first 5 rows of the dataframe") 
    print(df.head(5))
    
    #to show dimansions
    print(df.shape)
    
    #all information about a DataFrame
    print(df.info())
    
    #some statistica details
    print(df.describe())
    
    #missing data, False - no missing data, True - missing data
    missing_data = df.isnull()
    print(missing_data.head(5))
    
    #Count missing values in each column
    for column in missing_data.columns.values.tolist():
        print(column)
        print (missing_data[column].value_counts())
        print("")    
        
    #visualization    
    labels= 'Diabetic','Not Diabetic'
    plt.pie(df['Outcome'].value_counts(),labels=labels,autopct='%0.02f%%')
    plt.legend()
    plt.show()
        
    
    

def main():
    #csv_file_reading()
    #json_file_reading_writing()
    #xlsx_file_reading()
    #xml_file_reading()
    #image_file_reading()
    dataset_analysis()




if __name__ == "__main__": 
    t1=time.perf_counter()
    main()
    t2=time.perf_counter()
    print(f'Finished in {t2-t1} seconds')