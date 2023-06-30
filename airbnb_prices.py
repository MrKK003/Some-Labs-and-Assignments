import time
import numpy as np
import pandas as pd 
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
import xgboost as xgb
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler,PolynomialFeatures
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_selection import SequentialFeatureSelector,SelectFromModel
from sklearn.linear_model import LinearRegression,Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error,r2_score
import warnings

def combine(csv_1,col_1,csv_2,col_2,city):       # Combine the 2 datasets into a single dataset.
    csv_1['week time'] = col_1
    csv_2['week time'] = col_2
    csv_1.drop(columns = ['Unnamed: 0'],inplace=True)
    csv_2.drop(columns = ['Unnamed: 0'],inplace=True)
    merged = pd.concat([csv_1, csv_2])
    merged['city'] = city
    return merged

def time_of_the_week_effect(europe_data_2):
    plt.figure
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(15, 3))
    fig2, axs2 = plt.subplots(nrows=1, ncols=1, figsize=(15, 3))
    sns.boxplot(y='realSum', data=europe_data_2,x='week time',ax = axs[0])
    axs[0].tick_params(axis='y', labelsize=15)
    axs[0].tick_params(axis='x', labelsize=15)

    europe_data_2.groupby('week time')['realSum'].plot(kind='hist', alpha=0.15, bins=15,ax=axs[1])

    sns.kdeplot(data=europe_data_2[europe_data_2['week time'] == 'weekdays']['realSum'], label='weekdays',ax=axs2)
    sns.kdeplot(data=europe_data_2[europe_data_2['week time'] == 'weekends']['realSum'], label='weekends',ax=axs2)
    plt.subplots_adjust(hspace=0.65)
    plt.show()

def sequential_feature_selection(model,X_train,Y_train,X_test):
    sfs = SequentialFeatureSelector(model,  direction='backward', scoring='r2', cv=5)
    sfs.fit(X_train, Y_train)
    X_train_selected = sfs.transform(X_train)
    X_test_selected = sfs.transform(X_test)
    return X_train_selected, X_test_selected

def combine_lat_long(lng, lat):
    latitude = np.radians(lat)
    longitude = np.radians(lng)

    amsterdam_latitude = np.radians(0)
    amsterdam_longitude = np.radians(0)

    # apply Haversine formula to compute distance
    latitude_distance = amsterdam_latitude - latitude
    longitude_distance = amsterdam_longitude - longitude
    a = np.sin(latitude_distance/2)**2 + np.cos(latitude) * np.cos(amsterdam_latitude) * np.sin(longitude_distance/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    distance = 6371 * c

    return distance

def main():
    amsterdam_weekdays = pd.read_csv('/Applications/Some-Labs-and-Assignments/kaggle/input/airbnb-prices-in-european-cities/amsterdam_weekdays.csv')
    amsterdam_weekends = pd.read_csv('/Applications/Some-Labs-and-Assignments/kaggle/input/airbnb-prices-in-european-cities/amsterdam_weekends.csv')
    athens_weekdays = pd.read_csv('/Applications/Some-Labs-and-Assignments/kaggle/input/airbnb-prices-in-european-cities/athens_weekdays.csv')
    athens_weekends = pd.read_csv('/Applications/Some-Labs-and-Assignments/kaggle/input/airbnb-prices-in-european-cities/athens_weekends.csv')
    barcelona_weekdays = pd.read_csv('/Applications/Some-Labs-and-Assignments/kaggle/input/airbnb-prices-in-european-cities/barcelona_weekdays.csv')
    barcelona_weekends = pd.read_csv('/Applications/Some-Labs-and-Assignments/kaggle/input/airbnb-prices-in-european-cities/barcelona_weekends.csv')
    berlin_weekdays = pd.read_csv('/Applications/Some-Labs-and-Assignments/kaggle/input/airbnb-prices-in-european-cities/berlin_weekdays.csv')
    berlin_weekends = pd.read_csv('/Applications/Some-Labs-and-Assignments/kaggle/input/airbnb-prices-in-european-cities/berlin_weekends.csv')
    budapest_weekdays = pd.read_csv('/Applications/Some-Labs-and-Assignments/kaggle/input/airbnb-prices-in-european-cities/budapest_weekdays.csv')
    budapest_weekends = pd.read_csv('/Applications/Some-Labs-and-Assignments/kaggle/input/airbnb-prices-in-european-cities/budapest_weekends.csv')
    lisbon_weekdays = pd.read_csv('/Applications/Some-Labs-and-Assignments/kaggle/input/airbnb-prices-in-european-cities/lisbon_weekdays.csv')
    lisbon_weekends = pd.read_csv('/Applications/Some-Labs-and-Assignments/kaggle/input/airbnb-prices-in-european-cities/lisbon_weekends.csv')
    london_weekdays = pd.read_csv('/Applications/Some-Labs-and-Assignments/kaggle/input/airbnb-prices-in-european-cities/london_weekdays.csv')
    london_weekends = pd.read_csv('/Applications/Some-Labs-and-Assignments/kaggle/input/airbnb-prices-in-european-cities/london_weekends.csv')
    paris_weekdays = pd.read_csv('/Applications/Some-Labs-and-Assignments/kaggle/input/airbnb-prices-in-european-cities/paris_weekdays.csv')
    paris_weekends = pd.read_csv('/Applications/Some-Labs-and-Assignments/kaggle/input/airbnb-prices-in-european-cities/paris_weekends.csv')
    rome_weekdays = pd.read_csv('/Applications/Some-Labs-and-Assignments/kaggle/input/airbnb-prices-in-european-cities/rome_weekdays.csv')
    rome_weekends = pd.read_csv('/Applications/Some-Labs-and-Assignments/kaggle/input/airbnb-prices-in-european-cities/rome_weekends.csv')
    vienna_weekdays = pd.read_csv('/Applications/Some-Labs-and-Assignments/kaggle/input/airbnb-prices-in-european-cities/vienna_weekdays.csv')
    vienna_weekends = pd.read_csv('/Applications/Some-Labs-and-Assignments/kaggle/input/airbnb-prices-in-european-cities/vienna_weekends.csv')
    
    amsterdam = combine(amsterdam_weekdays,'weekdays',amsterdam_weekends,'weekends','amsterdam')
    athens = combine(athens_weekdays,'weekdays',athens_weekends,'weekends','athens')
    barcelona = combine(barcelona_weekdays,'weekdays',barcelona_weekends,'weekends','barcelona')
    berlin = combine(berlin_weekdays,'weekdays',berlin_weekends,'weekends','berlin')
    budapest = combine(budapest_weekdays,'weekdays',budapest_weekends,'weekends','budapest')
    lisbon = combine(lisbon_weekdays,'weekdays',lisbon_weekends,'weekends','lisbon')
    london = combine(london_weekdays,'weekdays',london_weekends,'weekends','london')
    paris = combine(paris_weekdays,'weekdays',paris_weekends,'weekends','paris')
    rome = combine(rome_weekdays,'weekdays',rome_weekends,'weekends','rome')
    vienna = combine(vienna_weekdays,'weekdays',vienna_weekends,'weekends','vienna')
    
    cities_names = ['amsterdam', 'athens', 'barcelona', 'berlin', 'budapest', 'lisbon', 'london', 'paris', 'rome', 'vienna']
    cities = [amsterdam, athens, barcelona, berlin, budapest, lisbon, london, paris, rome, vienna]
    europe_data = pd.concat(cities, ignore_index=True) 
    '''
    print(europe_data.head())
    print(europe_data.info())
    print(europe_data.describe())
    '''
    
    cities_2 = [amsterdam[amsterdam['realSum'] < 2000], athens[athens['realSum'] < 500], barcelona[barcelona['realSum'] < 1000], berlin[berlin['realSum'] < 800], budapest[budapest['realSum'] < 550], lisbon[lisbon['realSum'] < 650], london[london['realSum'] < 1500], paris[paris['realSum'] < 1200], rome[rome['realSum'] < 550], vienna[vienna['realSum'] < 750]]
    europe_data_2 = pd.concat(cities_2, ignore_index=True)
    
    
    europe_data_2_numerical_features = list(europe_data_2.select_dtypes(include=['int64','float64']).columns[i] for i in [1,4,5,6,7,8,9,10,11,12,13,14])
    
    europe_data_2_categorical_features = ['room_type','room_shared','room_private','host_is_superhost','multi','biz','week time']
    
    #DATA PRE-PROCESSING
    
    europe_data_2.replace({False: 0, True: 1},inplace=True)
    
    europe_data_2_categorical_dummies = pd.get_dummies(europe_data_2[['room_type','week time','city']],drop_first=True)
    
    europe_data_3 = pd.concat([europe_data_2_categorical_dummies, europe_data_2.drop(columns=['room_type','week time', 'city'])], axis=1)
    
    europe_data_3.drop(columns = ['rest_index_norm','attr_index_norm','room_shared','room_private'],inplace=True)
    
    europe_data_3['Haversine Distance'] = combine_lat_long(europe_data_3['lng'],europe_data_3['lat'])
    
    europe_data_3.drop(columns=['lng','lat'],inplace=True)
    
    #SCALING THE FEATURES
    
    Standard_Scaler = StandardScaler()
    features_to_scale = ['person_capacity','cleanliness_rating','guest_satisfaction_overall','bedrooms','dist','metro_dist','attr_index','rest_index','Haversine Distance']
    features_not_to_scale = ['room_type_Private room', 'room_type_Shared room', 'week time_weekends','city_athens', 'city_barcelona', 'city_berlin', 'city_budapest','city_lisbon', 'city_london', 'city_paris', 'city_rome', 'city_vienna','realSum','host_is_superhost', 'multi', 'biz',]
    
    scaled_features = pd.DataFrame(Standard_Scaler.fit_transform(europe_data_3[features_to_scale]), columns=features_to_scale)
    europe_data_final = pd.concat([scaled_features.reset_index(drop=True),  europe_data_3[features_not_to_scale].reset_index(drop=True)], axis=1)
    
    #REGRESSION MODELS
    
    X_train , X_test , Y_train , Y_test = train_test_split(europe_data_final.drop(columns=['realSum']) , europe_data_final['realSum'],random_state=4,test_size=0.15,stratify=europe_data_final[['week time_weekends','city_athens', 'city_barcelona', 'city_berlin', 'city_budapest','city_lisbon', 'city_london', 'city_paris', 'city_rome', 'city_vienna']])
    
    #LINEAR REGRESSION
    LR = LinearRegression()
    LR.fit(X_train,Y_train)
    LR_fit_evaulation = {'Linear Regression Fitting Evaluation':
        {'No. of Features':LR.n_features_in_,
         'R_squared score (Train)':LR.score(X_train,Y_train),
         'R_squared score (Test)':LR.score(X_test,Y_test)}}
    
    
    LR_fit_evaulation = pd.DataFrame(LR_fit_evaulation)
    print(LR_fit_evaulation)
    
    LR_TrainSet_Prediction = LR.predict(X_train)
    LR_TestSet_Prediction = LR.predict(X_test)
    
    LR_predict_evaulation = {'Linear Regression Predictions Evaluation':
    {'Train MSE': mean_squared_error(Y_train,LR_TrainSet_Prediction),
     'Test MSE' : mean_squared_error(Y_test,LR_TestSet_Prediction),
     'Train RMSE': mean_squared_error(Y_train,LR_TrainSet_Prediction,squared=False),
     'Test RMSE' : mean_squared_error(Y_test,LR_TestSet_Prediction,squared=False)}}
    
    LR_predict_evaulation = pd.DataFrame(LR_predict_evaulation)
    
    print(LR_predict_evaulation)
    
    #RIDGE REGRESSION
    R = Ridge()
    R.fit(X_train,Y_train)
    R_fit_evaulation = {'Ridge Regression Fitting Evaluation':
    {'No. of Features':R.n_features_in_,
     'R_squared score (Train)':R.score(X_train,Y_train),
     'R_squared score (Test)':R.score(X_test,Y_test)}}
    

    R_fit_evaulation = pd.DataFrame(R_fit_evaulation)
    print(R_fit_evaulation)
    
    R_TrainSet_Prediction = R.predict(X_train)
    R_TestSet_Prediction = R.predict(X_test)
    
    R_predict_evaulation = {'Ridge Regression Predictions Evaluation':
    {'Train MSE': mean_squared_error(Y_train,R_TrainSet_Prediction),
     'Test MSE' : mean_squared_error(Y_test,R_TestSet_Prediction),
     'Train RMSE': mean_squared_error(Y_train,R_TrainSet_Prediction,squared=False),
     'Test RMSE' : mean_squared_error(Y_test,R_TestSet_Prediction,squared=False)}}
    
    R_predict_evaulation = pd.DataFrame(R_predict_evaulation)
    
    print(R_predict_evaulation)
    
    
    #DECISION TREE REGRESSION
    
    DTR = DecisionTreeRegressor()
    DTR.fit(X_train,Y_train)
    DTR_fit_evaulation = {'Decision Tree Regression Fitting Evaluation':
    {'No. of Features':DTR.n_features_in_,
     'R_squared score (Train)':DTR.score(X_train,Y_train),
     'R_squared score (Test)':DTR.score(X_test,Y_test)}}
    DTR_fit_evaulation = pd.DataFrame(DTR_fit_evaulation)
    
    print(DTR_fit_evaulation)
    
    DTR_TrainSet_Prediction = DTR.predict(X_train)
    DTR_TestSet_Prediction = DTR.predict(X_test)
    
    DTR_predict_evaulation = {'Decision Tree Regression Predictions Evaluation':
    {'Train MSE': mean_squared_error(Y_train,DTR_TrainSet_Prediction),
     'Test MSE' : mean_squared_error(Y_test,DTR_TestSet_Prediction),
     'Train RMSE': mean_squared_error(Y_train,DTR_TrainSet_Prediction,squared=False),
     'Test RMSE' : mean_squared_error(Y_test,DTR_TestSet_Prediction,squared=False)}}
    
    DTR_predict_evaulation = pd.DataFrame(DTR_predict_evaulation)
    print(DTR_predict_evaulation)
    
    #RANDOM FOREST REGRESSION
    
    RFR = RandomForestRegressor(n_estimators = 50,max_depth=60)
    
    RFR.fit(X_train,Y_train)
    
    RFR_fit_evaulation = {'Random Forest Regression Fitting Evaluation (All Features)':
    {'No. of Features':RFR.n_features_in_,
     'R_squared score (Train)':RFR.score(X_train,Y_train),
     'R_squared score (Test)':RFR.score(X_test,Y_test)}}
    
    RFR_fit_evaulation = pd.DataFrame(RFR_fit_evaulation)
    print(RFR_fit_evaulation)
    
    RFR_TrainSet_Prediction = RFR.predict(X_train)
    RFR_TestSet_Prediction = RFR.predict(X_test)
    
    RFR_predict_evaulation = {'Ranndom Forest Regression Predictions Evaluation ':
    {'Train MSE': mean_squared_error(Y_train,RFR_TrainSet_Prediction),
     'Test MSE' : mean_squared_error(Y_test,RFR_TestSet_Prediction),
     'Train RMSE': mean_squared_error(Y_train,RFR_TrainSet_Prediction,squared=False),
     'Test RMSE' : mean_squared_error(Y_test,RFR_TestSet_Prediction,squared=False)}}
    
    RFR_predict_evaulation = pd.DataFrame(RFR_predict_evaulation)
    print(RFR_predict_evaulation)
    
    
    
    
    
if __name__ == "__main__": 
    t1=time.perf_counter()
    main()
    t2=time.perf_counter()
    print(f'Finished in {t2-t1} seconds')




