import pmdarima as pm
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

# Use auto ARIMA model
x = 0
for i in dataset_trend_array:
    x += 1
    model = pm.auto_arima(i, start_p=1, start_q=1,
                          test='adf',       # use adftest to find optimal 'd'
                          max_p=3, max_q=3, # maximum p and q
                          m=1,              # frequency of series
                          d=None,           # let model determine 'd'
                          seasonal=False,   # No Seasonality
                          start_P=0, 
                          D=0, 
                          trace=True,
                          error_action='ignore',  
                          suppress_warnings=True, 
                          stepwise=True)
    
    # print(model.summary())
    
    model.plot_diagnostics(figsize=(7,5))
    plt.show()
    
    # Forecast
    n_periods = 100
    fc, confint = model.predict(n_periods=n_periods, return_conf_int=True)
    index_of_fc = np.arange(len(i), len(i)+n_periods)
    
    # make series for plotting purpose
    fc_series = pd.Series(fc, index=index_of_fc)
    lower_series = pd.Series(confint[:, 0], index=index_of_fc)
    upper_series = pd.Series(confint[:, 1], index=index_of_fc)
    
    # Plot
    plt.plot(i)
    plt.plot(fc_series, color='darkgreen')
    plt.fill_between(lower_series.index, 
                      lower_series, 
                      upper_series, 
                      color='k', alpha=.15)
    
    plt.title("Final Forecast Top %s Popularity" %x)
    plt.show()
    
    # # Get the score for the trend
    # print('R2score is',r2_score(test, fc))
    # mean_absolute_percentage_error = np.mean(np.abs(mean - test)/np.abs(test))*100
    # print('MAPE is', mean_absolute_percentage_error)
    # print("\n")
    # print(forecast_accuracy(mean, test))