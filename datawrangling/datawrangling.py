import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def main():
    filename = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DA0101EN-SkillsNetwork/labs/Data%20files/auto.csv"
    
    headers = ["symboling","normalized-losses","make","fuel-type","aspiration", "num-of-doors","body-style",
         "drive-wheels","engine-location","wheel-base", "length","width","height","curb-weight","engine-type",
         "num-of-cylinders", "engine-size","fuel-system","bore","stroke","compression-ratio","horsepower",
         "peak-rpm","city-mpg","highway-mpg","price"]
    
    df = pd.read_csv(filename, names = headers)
    
    #print(df.to_string(buf=None, columns=None, col_space=None, header=True, index=True, na_rep='NaN', formatters=None, float_format=None, index_names=True, justify=None, max_rows=None, max_cols=None, show_dimensions=False, decimal='.', line_width=None))
    df.replace("?", np.nan, inplace = True)
    
    #print(df.to_string(buf=None, columns=None, col_space=None, header=True, index=True, na_rep='NaN', formatters=None, float_format=None, index_names=True, justify=None, max_rows=None, max_cols=None, show_dimensions=False, decimal='.', line_width=None))
    missing_data = df.isnull()    
    
    
    for column in missing_data.columns.values.tolist():
        print(column)
        print (missing_data[column].value_counts())
        print("")    
    
    avg_norm_loss = df["normalized-losses"].astype("float").mean(axis=0)
    #print("Average of normalized-losses:", avg_norm_loss)
    
    df["normalized-losses"].replace(np.nan, avg_norm_loss, inplace=True)
    #print(df.to_string(buf=None, columns=None, col_space=None, header=True, index=True, na_rep='NaN', formatters=None, float_format=None, index_names=True, justify=None, max_rows=None, max_cols=None, show_dimensions=False, decimal='.', line_width=None))
    
    avg_bore=df['bore'].astype('float').mean(axis=0)
    #print("Average of bore:", avg_bore)
    df["bore"].replace(np.nan, avg_bore, inplace=True)
    
    avg_stroke=df['stroke'].astype('float').mean(axis=0)
    #print("Average of bore:", avg_stroke)
    df["stroke"].replace(np.nan, avg_stroke, inplace=True)
    
    avg_horsepower = df['horsepower'].astype('float').mean(axis=0)
    #print("Average horsepower:", avg_horsepower)
    df['horsepower'].replace(np.nan, avg_horsepower, inplace=True)
    
    avg_peakrpm=df['peak-rpm'].astype('float').mean(axis=0)
    #print("Average peak rpm:", avg_peakrpm)
    df['peak-rpm'].replace(np.nan, avg_peakrpm, inplace=True)
    
    #print(df['num-of-doors'].value_counts())
    
    #print(df['num-of-doors'].value_counts().idxmax())
    
    df["num-of-doors"].replace(np.nan, "four", inplace=True)
    
    # drop whole row with NaN in "price" column
    df.dropna(subset=["price"], axis=0, inplace=True)

    # reset index, because we droped two rows
    df.reset_index(drop=True, inplace=True)
    
    dl=df.head()
    
    #print(df.to_string(buf=None, columns=None, col_space=None, header=True, index=True, na_rep='NaN', formatters=None, float_format=None, index_names=True, justify=None, max_rows=None, max_cols=None, show_dimensions=False, decimal='.', line_width=None)) #print whole dataframe
    
    #.dtype() to check the data type
    #.astype() to change the data type
    
    #print(df.dtypes)
    
    df[["bore", "stroke"]] = df[["bore", "stroke"]].astype("float")
    df[["normalized-losses"]] = df[["normalized-losses"]].astype("int")
    df[["price"]] = df[["price"]].astype("float")
    df[["peak-rpm"]] = df[["peak-rpm"]].astype("float")
    
    #print(df.dtypes)
    
    
    """Data Standartization"""
    
    
    # Convert mpg to L/100km by mathematical operation (235 divided by mpg)
    df['city-L/100km'] = 235/df["city-mpg"]
    
    df['highway-L/100km']= 235/df['highway-mpg']
    
    #print(df.head())
    
    
    """Data Normalization"""
    
    # replace (original value) by (original value)/(maximum value)
    df['length'] = df['length']/df['length'].max()
    df['width'] = df['width']/df['width'].max()
    df['height'] = df['height']/df['height'].max()
    
    
    """Binning"""
    
    
    df["horsepower"]=df["horsepower"].astype(int, copy=True)
    
    """ plt.hist(df["horsepower"])
    # set x/y labels and plot title
    plt.xlabel("horsepower")
    plt.ylabel("count")
    plt.title("horsepower bins")
    plt.show() """
    
    #linspace(start_value, end_value, numbers_generated)
    
    bins = np.linspace(min(df["horsepower"]), max(df["horsepower"]), 4)
    #print(bins)   
    group_names = ['Low', 'Medium', 'High']
    df['horsepower-binned'] = pd.cut(df['horsepower'], bins, labels=group_names, include_lowest=True )
    #print(df[['horsepower','horsepower-binned']].head(20))
    print(df["horsepower-binned"].value_counts())
    
    """ plt.bar(group_names, df["horsepower-binned"].value_counts())
    # set x/y labels and plot title
    plt.xlabel("horsepower")
    plt.ylabel("count")
    plt.title("horsepower bins")
    plt.show()
    """
    
    
    """Dummy variable"""
    
    dummy_variable_1 = pd.get_dummies(df["fuel-type"])
    dummy_variable_1.rename(columns={'gas':'fuel-type-gas', 'diesel':'fuel-type-diesel'}, inplace=True)
    #print(dummy_variable_1.head())
    
    # merge data frame "df" and "dummy_variable_1" 
    df = pd.concat([df, dummy_variable_1], axis=1)

    # drop original column "fuel-type" from "df"
    df.drop("fuel-type", axis = 1, inplace=True)
    
    dummy_variable_2 = pd.get_dummies(df["aspiration"])
    dummy_variable_2.rename(columns={'std':'aspiration-std', 'turbo':'aspiration-turbo'}, inplace=True)
    print(dummy_variable_2.head())
    
    df = pd.concat([df, dummy_variable_2], axis=1)
    
    df.drop("aspiration", axis = 1, inplace=True)
    
    print(df.head())
    
    df.to_csv('clean_df.csv')
    
    
    
    
    

    
if __name__ == "__main__": 
    t1=time.perf_counter()
    main()
    t2=time.perf_counter()
    print(f'Finished in {t2-t1} seconds')