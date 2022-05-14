import time
import requests
import os 
from PIL import Image
from IPython.display import IFrame

def http_request():
    #url='https://www.ibm.com/'
    url='https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-PY0101EN-SkillsNetwork/IDSNlogo.png'
    r=requests.get(url)
    
    #print(r.status_code)
    print(r.request.headers)
    #print("request body:", r.request.body)
    print(r.headers)
    
    path=os.path.join(os.getcwd(),'image.png') #get current working directory
    
    with open(path,'wb') as f:
        f.write(r.content)
         
    Image.open(path) 

def url_request():
    url_get='http://httpbin.org/get'
    payload={"name":"Joseph","ID":"123"}
    r=requests.get(url_get,params=payload)
    print(r.json())

def post_request():
    payload={"name":"Joseph","ID":"123"}
    url_post='http://httpbin.org/post'
    r_post=requests.post(url_post,data=payload)
    print(r_post.json()['form'])
    

def main():
    http_request()
    url_request()
    post_request()

    


if __name__ == "__main__": 
    t1=time.perf_counter()
    main()
    t2=time.perf_counter()
    print(f'Finished in {t2-t1} seconds')