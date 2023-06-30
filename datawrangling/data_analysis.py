import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


def main():
    path='https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DA0101EN-SkillsNetwork/labs/Data%20files/automobileEDA.csv'
    df = pd.read_csv(path)
    
    #print(df.dtypes)
    #print(df.corr())
    #print(df[['bore','stroke','compression-ratio','horsepower']].corr())
    
    # Engine size as potential predictor variable of price
    sns.regplot(x="engine-size", y="price", data=df)
    plt.ylim(0,)
    plt.xlim(0,)
    plt.show()
    
    #print(df[["engine-size", "price"]].corr())
    
    """sns.regplot(x="highway-mpg", y="price", data=df)
    plt.show()"""
    
    #print(df[['highway-mpg', 'price']].corr())
    
    """
    sns.regplot(x="peak-rpm", y="price", data=df)
    plt.show()"""
    
    #print(df[['peak-rpm', 'price']].corr())
    
    #print(df[['stroke', 'price']].corr())
    
    #sns.boxplot(x="body-style", y="price", data=df)
    
    #sns.boxplot(x="engine-location", y="price", data=df)
    
    # drive-wheels
    #sns.boxplot(x="drive-wheels", y="price", data=df)
    
    #plt.show()
    
    
    """Value Counts"""
    #print(df.describe(include=['object']))

    # engine-location as variable
    engine_loc_counts = df['engine-location'].value_counts().to_frame()
    engine_loc_counts.rename(columns={'engine-location': 'value_counts'}, inplace=True)
    engine_loc_counts.index.name = 'engine-location'
    print(engine_loc_counts)
    
    
    """Basic Grouping"""
    
    print(df['drive-wheels'].unique())
    
    df_group_one = df[['drive-wheels','body-style','price']]
    df_group_one = df_group_one.groupby(['drive-wheels'],as_index=False).mean()
    print(df_group_one)
    
    df_gptest = df[['drive-wheels','body-style','price']]
    grouped_test1 = df_gptest.groupby(['drive-wheels','body-style'],as_index=False).mean()
    #print(grouped_test1)
    
    grouped_pivot = grouped_test1.pivot(index='drive-wheels',columns='body-style')
    grouped_pivot = grouped_pivot.fillna(0) #fill missing values with 0
    print(grouped_pivot)
    
    #use the grouped results
    
    fig, ax = plt.subplots()
    im = ax.pcolor(grouped_pivot, cmap='RdBu')

    #label names
    row_labels = grouped_pivot.columns.levels[1]
    col_labels = grouped_pivot.index

    #move ticks and labels to the center
    ax.set_xticks(np.arange(grouped_pivot.shape[1]) + 0.5, minor=False)
    ax.set_yticks(np.arange(grouped_pivot.shape[0]) + 0.5, minor=False)

    #insert labels
    ax.set_xticklabels(row_labels, minor=False)
    ax.set_yticklabels(col_labels, minor=False)

    #rotate label if too long
    plt.xticks(rotation=90)

    fig.colorbar(im)
    #plt.show()
    
    
    
    """Correlation and Causation"""
    
    #Correlation: a measure of the extent of interdependence between variables.
    #Causation: the relationship between cause and effect between two variables.
    
    pearson_coef, p_value = stats.pearsonr(df['wheel-base'], df['price'])
    print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)  
    
    pearson_coef, p_value = stats.pearsonr(df['horsepower'], df['price'])
    print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P = ", p_value)  
    
    """ANOVA: Analysis of Variance"""
    
    grouped_test2=df_gptest[['drive-wheels', 'price']].groupby(['drive-wheels'])
    #print(grouped_test2.head(2))
    #print(grouped_test2.get_group('4wd')['price'])
    
    # ANOVA
    f_val, p_val = stats.f_oneway(grouped_test2.get_group('fwd')['price'], grouped_test2.get_group('rwd')['price'], grouped_test2.get_group('4wd')['price'])  
    print( "ANOVA results: F=", f_val, ", P =", p_val)  
    
    f_val, p_val = stats.f_oneway(grouped_test2.get_group('fwd')['price'], grouped_test2.get_group('rwd')['price'])  
    print( "ANOVA results: F=", f_val, ", P =", p_val )
    
    f_val, p_val = stats.f_oneway(grouped_test2.get_group('4wd')['price'], grouped_test2.get_group('rwd')['price'])  
    print( "ANOVA results: F=", f_val, ", P =", p_val)   
    
    f_val, p_val = stats.f_oneway(grouped_test2.get_group('4wd')['price'], grouped_test2.get_group('fwd')['price'])  
    print("ANOVA results: F=", f_val, ", P =", p_val)
    
    


    
    
    
    
    
    
    
    
    
    
    
    


if __name__ == "__main__": 
    t1=time.perf_counter()
    main()
    t2=time.perf_counter()
    print(f'Finished in {t2-t1} seconds')