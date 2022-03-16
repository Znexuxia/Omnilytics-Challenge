# Import any important package to the file
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import acf
from statsmodels.tsa.stattools import adfuller
import pmdarima as pm
from scipy.stats import skew
from sklearn.metrics import r2_score
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import pylab as p 
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings
import pickle
warnings.filterwarnings("ignore")


# Retrieve the data from the file
data_trend = "trends1_cleaned.csv"
dataset_trend_month = pd.read_csv(data_trend,parse_dates=['date'],index_col='date',dayfirst=(True))
dataset_trend = pd.read_csv(data_trend,parse_dates=['date'],index_col='observe')

dataset_trend_array = [dataset_trend.top1, dataset_trend.top2, dataset_trend.top3]

data_predict = "predictions_cleaned.csv"
dataset_predict = pd.read_csv(data_predict,parse_dates=['date'],index_col='date',dayfirst=(True))

# Make an mean for each month value
data_month_trend = dataset_trend_month.resample('M').mean().astype("int")
data_month_predict = dataset_predict.resample('M').mean().astype("int")

# Summarize the Dataset Trend
# shape
# print(dataset_trend.shape)

# Statistical Summary
# descriptions
# print(dataset_trend.describe())

# print(dataset_trend)


# Separate each top for actual value
array_trend = data_month_trend.values
actual_top1 = array_trend[:,0]
actual_top2 = array_trend[:,1]
actual_top3 = array_trend[:,2]

# Array actual value
actual_array = [actual_top1, actual_top2, actual_top3]


# Separate each top for prediction value
array_predict = data_month_predict.values

# Get means for each top predict value
predict_top1 = array_predict[:,0:3].mean(axis=1).astype("int")
predict_top2 = array_predict[:,3:6].mean(axis=1).astype("int")
predict_top3 = array_predict[:,6:9].mean(axis=1).astype("int")

# Array predicted value
predict_array = [predict_top1, predict_top2, predict_top3]

# Question 1 Answer
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

x = 0
# get the accuracy result to check the performance for each top predicted value
for a , b in zip(actual_array , predict_array):
    
    
    accuracy = accuracy_score(a, b)*100
    x += 1
    print("top %s accuracy is %.2f percent" %(x, accuracy))
    print(forecast_accuracy(b, a))
    print("")

    
    
