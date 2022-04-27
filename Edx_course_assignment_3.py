import time
import pandas as pd




def main():
    print('Hi!')
    #Define a dictionary 'x'

    x = {'Name': ['Rose','John', 'Jane', 'Mary'], 'ID': [1, 2, 3, 4], 'Department': ['Architect Group', 'Software Group', 'Design Team', 'Infrastructure'], 
        'Salary':[100000, 80000, 50000, 60000]}
    
    # a = {'Student':['David', 'Samuel', 'Terry', 'Evan'],
    #  'Age':['27', '24', '22', '32'],
    #  'Country':['UK', 'Canada', 'China', 'USA'],
    #  'Course':['Python','Data Structures','Machine Learning','Web Development'],
    #  'Marks':['85','72','89','76']}
    
    #casting the dictionary to a DataFrame
    df = pd.DataFrame(x)
    # df2 = pd.DataFrame(a)

    print(df)
    
    # x = df[['ID']]
    # print(x)
    # z = df[['Department','Salary','ID']]
    # print(z)
    
    print(df.iloc[0, 0])
    # df2=df
    # df2=df2.set_index("Name")
    # print(df2)
    # print(df2.head())
    #print(df2.loc['Jane', 'Salary'])
    #print(df.iloc[0:2, 0:3])
    print(df.loc[0:2,'ID':'Department'])
    print(df.loc[2:3,'Name':'Department'])

    

if __name__ == "__main__": 
    t1=time.perf_counter()
    main()
    t2=time.perf_counter()
    print(f'Finished in {t2-t1} seconds')