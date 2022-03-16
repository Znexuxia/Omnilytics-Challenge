import warnings
import pandas as pd
import numpy as np
import statistics as st
from scipy.stats import gmean
from scipy.stats import sem
from scipy.stats import skew
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

# Retrieve the data from the file
data_predict = "predictions_cleaned_question2.csv"
dataset_predict = pd.read_csv(data_predict,parse_dates=['date'],dayfirst=(True))

# Get top 1 data
dataset_top1_fit1 = dataset_predict.fit_top1
dataset_top1_fit2 = dataset_predict.fit2_top1
dataset_top1_fit3 = dataset_predict.fit3_top1

# Get top 2 data
dataset_top2_fit1 = dataset_predict.fit_top2
dataset_top2_fit2 = dataset_predict.fit2_top2
dataset_top2_fit3 = dataset_predict.fit3_top2

# Get top 3 data
dataset_top3_fit1 = dataset_predict.fit_top3
dataset_top3_fit2 = dataset_predict.fit2_top3
dataset_top3_fit3 = dataset_predict.fit3_top3

# Get Next 3 Months after actual data
# top 1
dataset_top1_fit1_3M = dataset_top1_fit1.values[262:354]
dataset_top1_fit2_3M = dataset_top1_fit2.values[262:354]
dataset_top1_fit3_3M = dataset_top1_fit3.values[262:354]

# top 2
dataset_top2_fit1_3M = dataset_top2_fit1.values[262:354]
dataset_top2_fit2_3M = dataset_top2_fit2.values[262:354]
dataset_top2_fit3_3M = dataset_top2_fit3.values[262:354]

# top 3
dataset_top3_fit1_3M = dataset_top3_fit1.values[262:354]
dataset_top3_fit2_3M = dataset_top3_fit2.values[262:354]
dataset_top3_fit3_3M = dataset_top3_fit3.values[262:354]

# Get Next 6 Months after actual data
# top 1
dataset_top1_fit1_6M = dataset_top1_fit1.values[262:444]
dataset_top1_fit2_6M = dataset_top1_fit2.values[262:444]
dataset_top1_fit3_6M = dataset_top1_fit3.values[262:444]

# top 2
dataset_top2_fit1_6M = dataset_top2_fit1.values[262:444]
dataset_top2_fit2_6M = dataset_top2_fit2.values[262:444]
dataset_top2_fit3_6M = dataset_top2_fit3.values[262:444]

# top 3
dataset_top3_fit1_6M = dataset_top3_fit1.values[262:444]
dataset_top3_fit2_6M = dataset_top3_fit2.values[262:444]
dataset_top3_fit3_6M = dataset_top3_fit3.values[262:444]

# Get Next 12 Months after actual data
# top 1
dataset_top1_fit1_12M = dataset_top1_fit1.values[262:627]
dataset_top1_fit2_12M = dataset_top1_fit2.values[262:627]
dataset_top1_fit3_12M = dataset_top1_fit3.values[262:627]

# top 2
dataset_top2_fit1_12M = dataset_top2_fit1.values[262:627]
dataset_top2_fit2_12M = dataset_top2_fit2.values[262:627]
dataset_top2_fit3_12M = dataset_top2_fit3.values[262:627]

# top 3
dataset_top3_fit1_12M = dataset_top3_fit1.values[262:627]
dataset_top3_fit2_12M = dataset_top3_fit2.values[262:627]
dataset_top3_fit3_12M = dataset_top3_fit3.values[262:627]

# Make an array
datasets = [dataset_top1_fit1_3M, dataset_top1_fit2_3M, dataset_top1_fit3_3M,
            dataset_top1_fit1_6M, dataset_top1_fit2_6M, dataset_top1_fit3_6M,
            dataset_top1_fit1_12M, dataset_top1_fit2_12M, dataset_top1_fit3_12M,
            dataset_top2_fit1_3M, dataset_top2_fit2_3M, dataset_top2_fit3_3M,
            dataset_top2_fit1_6M, dataset_top2_fit2_6M, dataset_top2_fit3_6M,
            dataset_top2_fit1_12M, dataset_top2_fit2_12M, dataset_top2_fit3_12M,
            dataset_top3_fit1_3M, dataset_top3_fit2_3M, dataset_top3_fit3_3M,
            dataset_top3_fit1_6M, dataset_top3_fit2_6M, dataset_top3_fit3_6M,
            dataset_top3_fit1_12M, dataset_top3_fit2_12M, dataset_top3_fit3_12M]

top = [1,1,1,1,1,1,1,1,1,
       2,2,2,2,2,2,2,2,2,
       3,3,3,3,3,3,3,3,3]

fit = [1,2,3,1,2,3,1,2,3,
       1,2,3,1,2,3,1,2,3,
       1,2,3,1,2,3,1,2,3,]

month = [3,3,3,6,6,6,12,12,12,
         3,3,3,6,6,6,12,12,12,
         3,3,3,6,6,6,12,12,12]

for a ,b ,c ,d in zip(datasets, top, fit, month):
    # Visualize the data
    plt.plot(a)
    plt.title('Time series Top %s in Fit %s for %s months' %(b ,c ,d))
    plt.ylabel('Popularity of the data')
    plt.xticks(rotation=45)
    plt.show()
    
    # Measure the central tendency
    mean = st.mean(a)
    mode = st.mode(a)
    median = st.median(a)
    harmonic_mean = st.harmonic_mean(a)
    median_low = st.median_low(a)
    median_high = st.median_high(a)
    geometric_mean = gmean(a)
    
    # Measure dispersion
    variance = st.variance(a)
    std_deviation = st.stdev(a)
    std_error = sem(a)
    
    # Measure of asymmetry
    skewness = skew(a)
    
    # Measure of variability
    data_range = np.ptp(a)
    percentile_25 = np.percentile(a, 25)
    percentile_50 = np.percentile(a, 50)
    percentile_75 = np.percentile(a, 75)
    percentile_90 = np.percentile(a, 90)
    
    result = {'mean':mean, 'mode':mode, 'median': median,'skewness': skewness, 
            'harmonic mean': harmonic_mean, 'median low':median_low, 'median high': median_high, 
            'geometric mean': geometric_mean, 'variance':variance, 'standard deviation': std_deviation,
            'standard error': std_error, '25th percentile': percentile_25, '50th percentile': percentile_50,
            '75th percentile': percentile_75, '90th percentile': percentile_90}
    
    print("Analysis of Top %s Fit %s next %s months" % (b,c,d))
    print(result)
    print("")
    
    