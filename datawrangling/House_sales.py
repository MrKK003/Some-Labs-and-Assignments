import time
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler,PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge



def main():
    file_name='https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DA0101EN-SkillsNetwork/labs/FinalModule_Coursera/data/kc_house_data_NaN.csv'
    df=pd.read_csv(file_name)
    
    #Question 1. Display the data types of each column using the function dtypes, then take a screenshot and submit it, include your code in the image.
    print(df.dtypes)
    
    ####
    
    #Question 2. Drop the columns "id"  and "Unnamed: 0" from axis 1 using the method drop(), then use the method describe() to obtain a statistical summary of the data. Take a screenshot and submit it, make sure the inplace parameter is set to True
    
    df.drop(labels=["id","Unnamed: 0"],axis=1, inplace=True)
    print(df.describe())
    
    ####
    """
    print("number of NaN values for the column bedrooms :", df['bedrooms'].isnull().sum())
    print("number of NaN values for the column bathrooms :", df['bathrooms'].isnull().sum())
    """
    mean=df['bedrooms'].mean()
    df['bedrooms'].replace(np.nan, mean, inplace=True)
    
    mean=df['bathrooms'].mean()
    df['bathrooms'].replace(np.nan,mean, inplace=True)
    
    '''
    print("number of NaN values for the column bedrooms :", df['bedrooms'].isnull().sum())
    print("number of NaN values for the column bathrooms :", df['bathrooms'].isnull().sum())
    '''
    
    #Question 3. Use the method value_counts to count the number of houses with unique floor values, use the method .to_frame() to convert it to a dataframe.
    
    df1=df['floors'].value_counts().to_frame()
    print(df1)
    
    ####
    
    # Question 4. Use the function boxplot in the seaborn library  to  determine whether houses with a waterfront view or without a waterfront view have more price outliers.
    
    sns.boxplot(x=df['waterfront'], y=df['price'])
    #plt.show()
    
    ####
    
    # Question 5. Use the function regplot in the seaborn library  to  determine if the feature sqft_above is negatively or positively correlated with price.
    
    sns.regplot(x=df['sqft_above'], y=df['price'], line_kws={'color':'green'})
    #plt.show()
    
    ####
    
    print(df.corr()['price'].sort_values())
    
    X = df[['long']]
    Y = df['price']
    lm = LinearRegression()
    lm.fit(X,Y)
    print(lm.score(X, Y))
    
    
    #Question  6. Fit a linear regression model to predict the 'price' using the feature 'sqft_living' then calculate the R^2. Take a screenshot of your code and the value of the R^2.
    
    X = df[['sqft_living']]
    lm.fit(X,Y)
    print(lm.score(X, Y))
    
    ####
    
    #Question 7.Fit a linear regression model to predict the 'price' using the list of features:
    
    features =["floors", "waterfront","lat" ,"bedrooms" ,"sqft_basement" ,"view" ,"bathrooms","sqft_living15","sqft_above","grade","sqft_living"]  
    
    X=df[features]
    lm.fit(X,Y)
    print(lm.score(X,Y))
    
    ####
    
    Input=[('scale',StandardScaler()),('polynomial', PolynomialFeatures(include_bias=False)),('model',LinearRegression())]
    
    #Question 8. Use the list to create a pipeline object to predict the 'price', fit the object using the features in the list features, and calculate the R^2.
    
    pipe=Pipeline(Input)
    X = X.astype(float)
    pipe.fit(X,Y)
    print(pipe.score(X,Y))
    
    #predicting the price 
    ypipe=pipe.predict(X)
    print(ypipe[0:4])
    
    ####
    
    features =["floors", "waterfront","lat" ,"bedrooms" ,"sqft_basement" ,"view" ,"bathrooms","sqft_living15","sqft_above","grade","sqft_living"]    
    X = df[features]
    Y = df['price']

    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.15, random_state=1)

    print("number of test samples:", x_test.shape[0])
    print("number of training samples:",x_train.shape[0])
    
    #Question 9. Create and fit a Ridge regression object using the training data, set the regularization parameter to 0.1, and calculate the R^2 using the test data.
    
    RigeModel=Ridge(alpha=1)
    RigeModel.fit(x_train,y_train)
    print(RigeModel.score(x_test,y_test))
    
    ####
    
    
    #Question 10. Perform a second order polynomial transform on both the training data and testing data. Create and fit a Ridge regression object using the training data, set the regularisation parameter to 0.1, and calculate the R^2 utilising the test data provided. Take a screenshot of your code and the R^2.
    
    pr=PolynomialFeatures(degree=2)
    
    x_train_pr=pr.fit_transform(x_train)
    x_test_pr=pr.fit_transform(x_test)
    
    RidgeModel=Ridge(alpha=1)
    RidgeModel.fit(x_train_pr,y_train)
    print(RidgeModel.score(x_test_pr,y_test))
    
    
    
    
    
    
    
    
if __name__ == "__main__": 
    t1=time.perf_counter()
    main()
    t2=time.perf_counter()
    print(f'Finished in {t2-t1} seconds')