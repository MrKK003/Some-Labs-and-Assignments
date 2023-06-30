
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

def PlotPolly(model, independent_variable, dependent_variabble, Name):
    x_new = np.linspace(15, 55, 100)
    y_new = model(x_new)

    plt.plot(independent_variable, dependent_variabble, '.', x_new, y_new, '-')
    plt.title('Polynomial Fit with Matplotlib for Price ~ Length')
    ax = plt.gca()
    ax.set_facecolor((0.898, 0.898, 0.898))
    fig = plt.gcf()
    plt.xlabel(Name)
    plt.ylabel('Price of Cars')

    plt.show()
    plt.close()



def main():
    path = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DA0101EN-SkillsNetwork/labs/Data%20files/automobileEDA.csv'
    df = pd.read_csv(path)
    
    
    """Simple Linear Regrassion"""
    
    
    lm = LinearRegression()
    X = df[['highway-mpg']]
    Y = df['price'] 
    lm.fit(X,Y)
    Yhat=lm.predict(X)
    #print(Yhat[0:5])
    print(lm.intercept_,lm.coef_)
    
    """lm1=LinearRegression()
    X=df[['engine-size']]
    Y=df['price']
    lm1.fit(X,Y)
    print(lm1.intercept_,lm1.coef_)"""
    print("---------------------")
    
    """Multiple Linear Regrassion"""   
    
    Z = df[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']]
    lm.fit(Z, df['price'])
    print(lm.intercept_,lm.coef_)
    
    lm2=LinearRegression()
    lm2.fit(df[['normalized-losses','highway-mpg']],df['price'])
    print(lm2.intercept_, lm2.coef_)
    print("---------------------")
    
    """Model Evaluation Using Visualization"""
    
    #Regression Plot
    
    width = 12
    height = 10
    '''plt.figure(figsize=(width, height))
    sns.regplot(x="highway-mpg", y="price", data=df)
    plt.ylim(0,)
    
    plt.figure(figsize=(width, height))
    sns.regplot(x="peak-rpm", y="price", data=df)
    plt.ylim(0,)
    #plt.show()'''
    
    print(df[['peak-rpm','highway-mpg','price']].corr())
    
    #Residual Plot
    '''plt.figure(figsize=(width, height))
    sns.residplot(df['highway-mpg'], df['price'])
    #plt.show()'''
    
    #We can see from this residual plot that the residuals are not randomly spread around the x-axis, leading us to believe that maybe a non-linear model is more appropriate for this data
    
    #Multiple Linear Regression
    '''
    Y_hat = lm.predict(Z)
    plt.figure(figsize=(width, height))

    ax1 = sns.distplot(df['price'], hist=False, color="r", label="Actual Value")
    sns.distplot(Y_hat, hist=False, color="b", label="Fitted Values" , ax=ax1)
    
    plt.title('Actual vs Fitted Values for Price')
    plt.xlabel('Price (in dollars)')
    plt.ylabel('Proportion of Cars')

    #plt.show()
    #plt.close()'''
    print("---------------------")
    
    
    
    """Polynomial Regression and Pipelines"""
    
    x = df['highway-mpg']
    y = df['price']
    # Here we use a polynomial of the 3rd order (cubic) 
    f = np.polyfit(x, y, 3)
    p = np.poly1d(f)
    print(p)
    #PlotPolly(p, x, y, 'highway-mpg')
    #print(np.polyfit(x, y, 3))
    
    f1=np.polyfit(x, y, 11)
    p1=np.poly1d(f1)
    print(p1)
    #PlotPolly(p1, x, y, 'highway-mpg')
    
    #Multivariate Polynomial function
    pr=PolynomialFeatures(degree=2)
    Z_pr=pr.fit_transform(Z)
    print(Z.shape)
    print(Z_pr.shape)
    
    
    """Pipeline"""
    
    
    #We create the pipeline by creating a list of tuples including the name of the model or estimator and its corresponding constructor. Then, we can normalize the data,  perform a transform and fit the model simultaneously.
    Input=[('scale',StandardScaler()), ('polynomial', PolynomialFeatures(include_bias=False)), ('model',LinearRegression())]
    pipe=Pipeline(Input)
    
    Z = Z.astype(float)
    pipe.fit(Z,y)
    
    ypipe=pipe.predict(Z)
    print(ypipe[0:4])
    print("---------------------")
    
    
    
    
    
    """Measures for In-Sample Evaluation"""
    
    
    #R-squared. R squared, also known as the coefficient of determination, is a measure to indicate how close the data is to the fitted regression line. The value of the R-squared is the percentage of variation of the response variable (y) that is explained by a linear model.
    #Mean Squared Error (MSE). The Mean Squared Error measures the average of the squares of errors. That is, the difference between actual value (y) and the estimated value (ŷ).
    
    #Model 1: Simple Linear Regression
    
    #highway_mpg_fit
    lm.fit(X, Y)
    # Find the R^2
    print('The R-square is: ', lm.score(X, Y))
    
    Yhat=lm.predict(X)
    print('The output of the first four predicted value is: ', Yhat[0:4])
    mse = mean_squared_error(df['price'], Yhat)
    print('The mean square error of price and predicted value is: ', mse)
    
    #Model 2: Multiple Linear Regression
    
    # fit the model 
    lm.fit(Z, df['price'])
    # Find the R^2
    print('The R-square is: ', lm.score(Z, df['price']))
    
    Y_predict_multifit = lm.predict(Z)
    print('The mean square error of price and predicted value using multifit is: ', \
      mean_squared_error(df['price'], Y_predict_multifit))
    
    
    #Model 3: Polynomial Fit
    
    r_squared = r2_score(y, p(x))
    print('The R-square value is: ', r_squared)
    print(mean_squared_error(df['price'], p(x)))
    print("---------------------")
    
    
    
    
    """Prediction and Decision Making"""
    
    
    new_input=np.arange(1, 100, 1).reshape(-1, 1)
    
    lm.fit(X, Y)
    
    yhat=lm.predict(new_input)
    print(yhat)
    plt.plot(new_input, yhat)
    plt.show()  
    
    
    
    
    
    



if __name__ == "__main__": 
    t1=time.perf_counter()
    main()
    t2=time.perf_counter()
    print(f'Finished in {t2-t1} seconds')