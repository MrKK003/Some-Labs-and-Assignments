import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ipywidgets import interact, interactive, fixed, interact_manual
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from tqdm import tqdm
from sklearn.model_selection import GridSearchCV


def DistributionPlot(RedFunction, BlueFunction, RedName, BlueName, Title):
    width = 12
    height = 10
    plt.figure(figsize=(width, height))

    ax1 = sns.distplot(RedFunction, hist=False, color="r", label=RedName)
    ax2 = sns.distplot(BlueFunction, hist=False, color="b", label=BlueName, ax=ax1)

    plt.title(Title)
    plt.xlabel('Price (in dollars)')
    plt.ylabel('Proportion of Cars')

    plt.show()
    plt.close()
    
def PollyPlot(xtrain, xtest, y_train, y_test, lr,poly_transform):
    width = 12
    height = 10
    plt.figure(figsize=(width, height))
    
    
    #training data 
    #testing data 
    # lr:  linear regression object 
    #poly_transform:  polynomial transformation object 
 
    xmax=max([xtrain.values.max(), xtest.values.max()])

    xmin=min([xtrain.values.min(), xtest.values.min()])

    x=np.arange(xmin, xmax, 0.1)


    plt.plot(xtrain.values[:,:], y_train, 'ro', label='Training Data')
    plt.plot(xtest.values[:,:], y_test, 'go', label='Test Data')
    plt.plot(x, lr.predict(poly_transform.fit_transform(x.reshape(-1, 1))), label='Predicted Function')
    plt.ylim([-10000, 60000])
    plt.ylabel('Price')
    plt.legend()
    plt.show()
    
    
def f(order, test_data, x_data, y_data):
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=test_data, random_state=0)
    pr = PolynomialFeatures(degree=order)
    x_train_pr = pr.fit_transform(x_train[['horsepower']])
    x_test_pr = pr.fit_transform(x_test[['horsepower']])
    poly = LinearRegression()
    poly.fit(x_train_pr,y_train)
    PollyPlot(x_train[['horsepower']], x_test[['horsepower']], y_train,y_test, poly, pr)


def main():
    path = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DA0101EN-SkillsNetwork/labs/Data%20files/module_5_auto.csv'
    df = pd.read_csv(path)
    df.to_csv('module_5_auto.csv')
    
    #only numeric data
    
    df=df._get_numeric_data()
    
    
    """Training and Testing"""
    
    
    y_data = df['price']
    x_data=df.drop('price',axis=1)
    
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.1, random_state=1)

    print("number of test samples :", x_test.shape[0])
    print("number of training samples:",x_train.shape[0])
    
    lre=LinearRegression()
    
    lre.fit(x_train[['horsepower']], y_train)
    
    print(lre.score(x_train[['horsepower']], y_train))
    
    print(lre.score(x_test[['horsepower']], y_test))
    
    
    """Cross-Validation Score"""
    
    
    Rcross = cross_val_score(lre, x_data[['horsepower']], y_data, cv=4)
    print("The mean of the folds are", Rcross.mean(), "and the standard deviation is" , Rcross.std())
    
    #You can also use the function 'cross_val_predict' to predict the output. The function splits up the data into the specified number of folds, with one fold for testing and the other folds are used for training.
    yhat = cross_val_predict(lre,x_data[['horsepower']], y_data,cv=4)
    #print(yhat)
    
    
    """Overfitting, Underfitting and Model Selection"""
    
    
    lr = LinearRegression()
    lr.fit(x_train[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']], y_train)
    
    yhat_train = lr.predict(x_train[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']])
    print(yhat_train[0:5])
    
    yhat_test = lr.predict(x_test[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']])
    print(yhat_test[0:5])
    
    Title = 'Distribution  Plot of  Predicted Value Using Training Data vs Training Data Distribution'
    #DistributionPlot(y_train, yhat_train, "Actual Values (Train)", "Predicted Values (Train)", Title)
    
    Title='Distribution  Plot of  Predicted Value Using Test Data vs Data Distribution of Test Data'
    #DistributionPlot(y_test,yhat_test,"Actual Values (Test)","Predicted Values (Test)",Title)
    
    #Comparing Figure 1 and Figure 2, it is evident that the distribution of the test data in Figure 1 is much better at fitting the data. This difference in Figure 2 is apparent in the range of 5000 to 15,000. This is where the shape of the distribution is extremely different. Let's see if polynomial regression also exhibits a drop in the prediction accuracy when analysing the test dataset.
    
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.45, random_state=0)
    
    pr = PolynomialFeatures(degree=5)
    x_train_pr = pr.fit_transform(x_train[['horsepower']])
    x_test_pr = pr.fit_transform(x_test[['horsepower']])
    
    poly = LinearRegression()
    poly.fit(x_train_pr, y_train)
    yhat = poly.predict(x_test_pr)
    
    print("Predicted values:", yhat[0:4])
    print("True values:", y_test[0:4].values)
    
    #PollyPlot(x_train[['horsepower']], x_test[['horsepower']], y_train, y_test, poly,pr)
    
    print(poly.score(x_train_pr, y_train))
    print(poly.score(x_test_pr, y_test))
    
    Rsqu_test = []
    order = [1, 2, 3, 4]
    for n in order:
        pr = PolynomialFeatures(degree=n)
        
        x_train_pr = pr.fit_transform(x_train[['horsepower']])
        
        x_test_pr = pr.fit_transform(x_test[['horsepower']])    
        
        lr.fit(x_train_pr, y_train)
        
        Rsqu_test.append(lr.score(x_test_pr, y_test))

    plt.plot(order, Rsqu_test)
    plt.xlabel('order')
    plt.ylabel('R^2')
    plt.title('R^2 Using Test Data')
    plt.text(3, 0.75, 'Maximum R^2 ')   
    #plt.show()
    
    """Task"""
    
    
    """Task"""
    #interact(f, order=(0, 6, 1), test_data=(0.05, 0.95, 0.05),x_data=x_data, y_data=y_data)
    
    
    """Ridge Regression"""
    
    pr=PolynomialFeatures(degree=2)
    x_train_pr=pr.fit_transform(x_train[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg','normalized-losses','symboling']])
    x_test_pr=pr.fit_transform(x_test[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg','normalized-losses','symboling']])
    
    RigeModel=Ridge(alpha=1)
    RigeModel.fit(x_train_pr, y_train)
    yhat = RigeModel.predict(x_test_pr)
    print('predicted:', yhat[0:4])
    print('test set :', y_test[0:4].values)
    
    Rsqu_test = []
    Rsqu_train = []
    dummy1 = []
    Alpha = 10 * np.array(range(0,1000))
    pbar = tqdm(Alpha)

    for alpha in pbar:
        RigeModel = Ridge(alpha=alpha) 
        RigeModel.fit(x_train_pr, y_train)
        test_score, train_score = RigeModel.score(x_test_pr, y_test), RigeModel.score(x_train_pr, y_train)
        
        pbar.set_postfix({"Test Score": test_score, "Train Score": train_score})

        Rsqu_test.append(test_score)
        Rsqu_train.append(train_score)
    
    #print(Rsqu_test)
    #print(Rsqu_train)
       
    width = 12
    height = 10
    plt.figure(figsize=(width, height))

    plt.plot(Alpha,Rsqu_test, label='validation data  ')
    plt.plot(Alpha,Rsqu_train, 'r', label='training Data ')
    plt.xlabel('alpha')
    plt.ylabel('R^2')
    plt.legend()
    #plt.show()
    
    
    
    """Grid Search"""
    
    
    
    parameters1= [{'alpha': [0.001,0.1,1, 10, 100, 1000, 10000, 100000, 100000]}]
    RR=Ridge()
    Grid1 = GridSearchCV(RR, parameters1,cv=4)
    Grid1.fit(x_data[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']], y_data)
    BestRR=Grid1.best_estimator_
    print(BestRR.score(x_test[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']], y_test))
    
    
    

 
 
 
    
if __name__ == "__main__": 
    t1=time.perf_counter()
    main()
    t2=time.perf_counter()
    print(f'Finished in {t2-t1} seconds')
    