# Import any important package to the file
import pandas as pd
import statistics as st
import numpy as np
from scipy.stats import gmean
from scipy.stats import skew
from scipy.stats import sem
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import warnings

warnings.filterwarnings("ignore")

# Retrieve the data from the file
data = "trends1_cleaned.csv"
dataset_trend = pd.read_csv(data,parse_dates=['date'],index_col='date',dayfirst=(True))

# Make an mean for each month value
data_month = dataset_trend.resample('M').mean().astype("int")
      
# Visualize the data
plt.plot(data_month)
plt.title('Trend of variables')
plt.ylabel('Quantity of the data')
plt.xticks(rotation=45)
plt.show()



# Make a class for calculation on last 12 months
class last_12m(object):
    # Get a last 12 months dataset
    data_last_12M = dataset_trend.last("12M")
    print(data_last_12M)
    
    # Visualize the data
    plt.plot(data_last_12M)
    plt.title('Time series data in last 12 months')
    plt.ylabel('Popularity of the data')
    plt.xticks(rotation=45)
    plt.show()
    
    # Separate each top as a datasets
    array = data_last_12M.values
    dataset_top1 = array[:,0]
    dataset_top2 = array[:,1]
    dataset_top3 = array[:,2]
    
    # Put the datasets in array
    datasets = [dataset_top1,dataset_top2,dataset_top3]
    
    x = 0
    result = []
    for i in datasets:
        x += 1
        name = "top %s" %x
        
        # Measure the central tendency
        mean = st.mean(i)
        mode = st.mode(i)
        median = st.median(i)
        harmonic_mean = st.harmonic_mean(i)
        median_low = st.median_low(i)
        median_high = st.median_high(i)
        geometric_mean = gmean(i)
        
        # Measure dispersion
        variance = st.variance(i)
        std_deviation = st.stdev(i)
        std_error = sem(i)
        
        # Measure of asymmetry
        skewness = skew(i)
        
        # Measure of variability
        data_range = np.ptp(i)
        percentile_25 = np.percentile(i, 25)
        percentile_50 = np.percentile(i, 50)
        percentile_75 = np.percentile(i, 75)
        percentile_90 = np.percentile(i, 90)
        
        
        # Check accuracy with 1,1,1 ARIMA model
        
        # 1,1,1 ARIMA Model
        model = ARIMA(i, order=(1,1,1))
        model_fit = model.fit()
        # print(model_fit.summary())
        
        # Forecast
        predict = model_fit.predict(dynamic=(False)) # 95% conf
        
        # Calculate trend accuracy
        accuracy = accuracy_score(i, predict.astype("int"))
        
        listed = [name, data_range, mean, mode,
                median, harmonic_mean, median_low,
                median_high, geometric_mean,
                skewness, variance,
                std_deviation, std_error,
                percentile_25, percentile_50, percentile_75, percentile_90, accuracy]
        
        result.append(listed)
    
    
    # print(result)
        
obj = last_12m()

# Create class to show it performance for last 12M
class perf_12M():
        def parse_dict(self):
            for i in range(len(obj.result)):
                return_dict = {}
                return_dict['trend'] = obj.result[i][0]
                return_dict['data range'] = obj.result[i][1]
                return_dict['mean'] = obj.result[i][2]
                return_dict['mode'] = obj.result[i][3]
                return_dict['median'] = obj.result[i][4]
                return_dict['harmonic mean'] = obj.result[i][5]
                return_dict['median low'] = obj.result[i][6]
                return_dict['median high'] = obj.result[i][7]
                return_dict['geometric mean'] = obj.result[i][8]
                return_dict['skewness'] = obj.result[i][9]
                return_dict['variance'] = obj.result[i][10]
                return_dict['standard deviation'] = obj.result[i][11]
                return_dict['standard error'] = obj.result[i][12]
                return_dict['25th percentile'] = obj.result[i][13]
                return_dict['50th percentile'] = obj.result[i][14]
                return_dict['75th percentile'] = obj.result[i][15]
                return_dict['90th percentile'] = obj.result[i][16]
                return_dict['accuracy'] = obj.result[i][17]
                print(return_dict)
                print("\n")
                # return return_dict
        

print(perf_12M().parse_dict()) 
        
        
        
    
    
    
