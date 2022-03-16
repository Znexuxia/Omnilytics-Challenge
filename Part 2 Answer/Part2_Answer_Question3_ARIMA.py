import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import acf
from scipy.stats import skew
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.metrics import accuracy_score
from statsmodels.tsa.arima.model import ARIMA
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

# Use manual ARIMA model
x = 0
for i in dataset_trend_array:
    x += 1
    # 1,1,1 ARIMA Model
    model = ARIMA(i, order=(1,1,1))
    model_fit = model.fit()
    # print("\n")
    print(model_fit.summary())
    # print("\n")
    
    # Actual vs Fitted
    result = model_fit.predict(dynamic=(False))
    
    # Line plots of time series
    fig, ax = plt.subplots(figsize=(6,3))
    fig.suptitle('Top %s Actual vs Predicted plot' %x)
    result.plot(ax=ax, label = "Predicted")
    i.plot(ax=ax, label = "Actual")
    plt.legend()
    plt.show()
    
    acc = accuracy_score(i, result.values.astype("int"))*100
    
    print("1,1,1 ARIMA Model for top %s accuracy is %.2f percent" %(x, acc))
    
    print(forecast_accuracy(result, i))
    print("\n")

# Use manual ARIMA model
x = 0
for i in dataset_trend_array:
    x += 1
    # Set validation size
    validation_size = 0.20
    a = int(len(i) - len(i)*validation_size)
    b = int(len(i)*validation_size)
    # Create Training and Test
    train = i[:a]
    test = i[a:]

    # Build Model
    # model = ARIMA(train, order=(3,2,1))  
    model = ARIMA(train, order=(1, 1, 1))  
    fitted = model.fit()  
    
    # Forecast
    fc= fitted.forecast(100, alpha=0.05)  # 95% conf
    conf = fitted.forecast(15, alpha=0.05)
    se = fitted.forecast(15, alpha=0.05)
    # Make as pandas series
    fc_series = pd.Series(fc, index=test.index)

    # Plot
    plt.figure(figsize=(12,5), dpi=100)
    plt.plot(train, label='training')
    plt.plot(test, label='actual')
    plt.plot(fc_series, label='forecast')
    plt.title('Top %s Forecast vs Actuals'%x)
    plt.legend(loc='upper left', fontsize=8)
    plt.show()
