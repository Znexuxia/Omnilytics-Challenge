from statsmodels.tsa.statespace.sarimax import SARIMAX
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import acf
from scipy.stats import skew
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings("ignore")


# Retrieve the data from the file
data_trend = "trends1_cleaned.csv"
dataset_trend_month = pd.read_csv(data_trend,parse_dates=['date'],index_col='date',dayfirst=(True))
dataset_trend = pd.read_csv(data_trend,parse_dates=['date'],index_col='observe')

dataset_trend_array = [dataset_trend.top1, dataset_trend.top2, dataset_trend.top3]

# Accuracy metrics
def forecast_accuracy(forecast, actual):
    mape = np.mean(np.abs(forecast - actual)/np.abs(actual))  # MAPE
    me = np.mean(forecast - actual)             # ME
    mae = np.mean(np.abs(forecast - actual))    # MAE
    mpe = np.mean((forecast - actual)/actual)   # MPE
    rmse = np.mean((forecast - actual)**2)**.5  # RMSE
    corr = np.corrcoef(forecast, actual)[0,1]   # corr
    skewness = skew(forecast)
    mins = np.amin(np.hstack([forecast[:,None], 
                              actual[:,None]]), axis=1)
    maxs = np.amax(np.hstack([forecast[:,None], 
                              actual[:,None]]), axis=1)
    minmax = 1 - np.mean(mins/maxs)             # minmax
    acf1 = acf(forecast-actual)[1]                      # ACF1
    return({'mape':mape, 'me':me, 'mae': mae,'skewness': skewness, 
            'mpe': mpe, 'rmse':rmse, 'acf1':acf1, 
            'corr':corr, 'minmax':minmax})

# Use SARIMAX model
for i in dataset_trend_array:
    
    #divide into train and validation set
    train = i[:int(0.85*(len(i)))]
    test = i[int(0.85*(len(i))):]
    
    
    model = SARIMAX(train,order=(1,1,0),seasonal_order=(0,1,1,12))
    results = model.fit()
    print("Results of SARIMAX on train")
    print(results.summary())
    
    # Validation
    results.plot_diagnostics(figsize=(16, 8))
    plt.savefig('modeldiagnostics')
    plt.show()
    
    # Checking prediction value with test
    forecast = results.get_forecast(steps=len(test))
    mean = forecast.predicted_mean
    conf_int = forecast.conf_int()
    d = mean.index
    plt.figure(figsize=(8,5))
    # Plot past  levels
    plt.plot(i.index, i, label='Oiginal', color = 'red')
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
    print("\n")
    print(forecast_accuracy(mean, test))