# Import any important package to the file
import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import r2_score
import pmdarima as pm
import numpy as np
import matplotlib.pyplot as plt

# Retrieve the data from the file
data_trend = "trends1_cleaned.csv"
dataset_trend = pd.read_csv(data_trend,parse_dates=['date'],index_col='date',dayfirst=(True))

# Make an mean for each month value
dataset_trend_avg = dataset_trend.resample('M').mean().astype("int")


# print(data_month.top1)

class last_12M(object):
    def __init__(self,datasets):

        
        # Auto Correlation
        fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(14,6))
        ax1 = plot_acf(datasets, lags=30, ax=ax1)
        # ax2 = plot_pacf(datasets, lags=5, ax=ax2,method='ywm')
        plt.show()
        
        # Stationary test
        result = adfuller(datasets)
        print('ADF Statistic: %f' % result[0])
        print('p-value: %f' % result[1])
        print('Critical Test Statistics Values:')
        for key, value in result[4].items():
            print('\t%s: %.3f' % (key, value))
            
        # Differentiation
        datasets_diff = datasets.diff().diff(12)
        datasets_diff.dropna(inplace=True)
        
        # Plot differenced data
        fig, ax = plt.subplots(figsize=(12,9))
        fig.suptitle('Line Plot of the Stationary Seasonal Time Series Data')
        datasets_diff.plot(ax=ax)
        plt.show()

        # Using ARIMA
        model = pm.auto_arima(datasets, d=1, D=1,seasonal=True, m=12, trend='c',start_p=0, start_q=0, max_order=6, test='adf', stepwise=True, trace=True)
        print(model.summary())
        
        
        #divide into train and validation set
        train = datasets[:int(0.85*(len(datasets)))]
        test = datasets[int(0.85*(len(datasets))):]
        
        #plotting the data
        train.plot()
        test.plot()
        
        # Use SARIMAX
        model = SARIMAX(train,order=(1,1,0),seasonal_order=(0,1,1,12))
        results = model.fit()
        print("Results of SARIMAX on train")
        print(results.summary())
        
        # Validation
        results.plot_diagnostics(figsize=(16, 8))
        plt.savefig('modeldiagnostics')
        plt.show()
        
        #Checking prediction value with test
        forecast = results.get_forecast(steps=len(test))
        mean = forecast.predicted_mean
        conf_int = forecast.conf_int()
        d = mean.index
        plt.figure(figsize=(8,5))
        # Plot past  levels
        plt.plot(datasets.index, datasets, label='Oiginal', color = 'red')
        # Prediction Mean
        plt.plot(d, mean, label='Forecasted', color= 'green')
        # Shade of space between  confidence intervals
        plt.fill_between(d, conf_int.iloc[:,0], conf_int.iloc[:,1],
        alpha=0.2, color= 'blue')
        # Plot legend 
        plt.legend()
        plt.savefig('p')
        plt.show()
        
        # Get the score for the trend
        print('R2score is',r2_score(test, mean))
        mean_absolute_percentage_error = np.mean(np.abs(mean - test)/np.abs(test))*100
        print('MAPE is', mean_absolute_percentage_error)
        

class perf_12M:
    # Get a last 12 months dataset without averaging the value
    data_last_12M = dataset_trend.last("12M")
    data_last_12M_array = [data_last_12M.top1 , data_last_12M.top2, data_last_12M.top3]
    
    # Get a last 12 months dataset with averaging the value
    data_last_12M_avg = dataset_trend_avg.last("12M")
    
    # Summarize the Dataset Trend
    # Statistical Summary
    # descriptions
    print(dataset_trend.describe())
    
    # Seasonal Decomposition for Seasonal Time Series
    result = seasonal_decompose(dataset_trend.top1, model='additive')
    result.plot()
    plt.show()
    
    # Line plots of time series
    fig, ax = plt.subplots(figsize=(6,3))
    fig.suptitle('Time Series Data')
    data_last_12M.plot(ax=ax)
    plt.show()
    
    #Historgram
    fig, ax = plt.subplots(figsize=(6,3))
    fig.suptitle('Histogram')
    data_last_12M.hist(ax=ax)
    plt.show()
    
    # Box and whisker plots
    fig, ax = plt.subplots(figsize=(6,3))
    fig.suptitle('Box and Whisker')
    data_last_12M.boxplot(ax=ax)
    plt.show()
    
    # Separate each top as a datasets
    array = data_last_12M.values
    dataset_top1 = array[:,0]
    dataset_top2 = array[:,1]
    dataset_top3 = array[:,2]
    
    # print(data_last_12M.index)
    # Check result for the 3 data
    for i in data_last_12M_array:
        result = last_12M(i)
    