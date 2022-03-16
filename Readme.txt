Data Science Challenge Omnilytics

Prepared By: Mohd Shahazureen Ikwan Bin Abdul Rahman

Explanation part
Part 1 Answer (Conclusion)

Q1 Answer
- Based on the calculation result based on the file Part1_Answer.py the best
  performance trend is top 1 by comparing the accuracy for each of the trend.
  Based on the visual line chart graph, the orange line(presume in my end) is
  Top 2 which has the worst accuracy with approximate 4.17%. Top 1 which the
  blue line (presume in my end) has the best accuracy with approximate 18.75%
  followed by top 3 the green line (presume in my end) with accuracy approximate
  14.58%. These accuracy are calculated with a 1,1,1 ARIMA Model to compare it.

Part 2 Answer (Conclusion)

Q1 Answer
- Based on the result get in file Part2_Answer_Question1.py the best performance
  is top 3 with an accuracy of 16.39% which I get on comparing the value between
  the actual value and the predicted value. Median Absolute Percentage Error (MAPE)
  is around 23.77% which is the accuracy for the next forcasted period is 76.23%
  comparing to the other two predicted top but the worst is top 2 with 0% accuracy
  with 272% MAPE whic the accuracy for next forcasted period is -172% which we can
  conclude this top 2 trend is underfit type of dataset. Do keep in mind that all
  3 top analysis is calculated after getting the mean of 3 fit by each top.


Q2 Answer
- Based on this question it need a lot of observation on the analyst result value.
  From what we can see here we can focus on the basic analysist result such as
  mean, mode, median and skew. In file Part2_Answer_Question2.py, we can compare
  each top based on their changes for 3, 6 and 12 months. We can only focus on mean
  for this case starting with top 1 as the mean has nothing drastically change which
  is the it range around from 17 to 19 which is a good datasets. As for top 2 mean
  is keep changing drastically and cannot now the range from where. It is not changing
  based on 3,6 and 12 months but on it fit datasets too, clearly this top 2 trend is
  the worst datasets (underfit). Lastly, top 3 has a good result based on it mean as we
  can now the mean is ranging from 31 to 37 but not as good as top 1 trend.

Q4 Answer
- In this case, I have make 3 type of prediction model which is
  * 1,1,1 Autoregressive Integrated Moving Average(ARIMA) Model Prediction
  * Auto ARIMA (I can't use to get the result value yet)
  * Seasonal Auto-Regressive Integrated Moving Average(SARIMAX)

1) 1,1,1 ARIMA Model Prediction
  - The result in this model is giving a great value as each top has their accuracy with
    top 1 has the highest of 18.77% and 16.21% MAPE (which 83.79% accurate on next forecast)
    but the lowest in forecast result which can be assumed an overfit dataset followed by top 3
    and of course the least result of top 2 with 11.49% accuracy but best forcasting result
    of 10.63% MAPE (89.37% accurate on the next forecast). The graph give an abnormal result
    might need to fix the code but still weird on this one.

2) Auto ARIMA
  - could not explain as i still haven't figured to to calculate the result value. It will
    be good for future reference.

3) SARIMAX
  - In this prediction model we are focusing on the r2score(proportion of the variance in the
    dependent variable that is predictable from the independent variable) whether the predicted
    is correlated with the actual value. Before starting on use this model we start with Out
    Of Time Cross Validation which we choose a certain of period as a validation set and the rest of it
    as a train set. The best trend in this model is top 2 with -98.65% r2score and 22.57% MAPE which
    77.43% accurate on next forecast. The worst is top 1 but in this model all 3 top give a bad result.
    This is not a suitable model to use in this case.

- Based on the 2 models result I say that 1,1,1 ARIMA model is the best
- By comparing between the predicted trend and my prediction model, I can conclude that my model prediction
  for 1,1,1 ARIMA model perform better than the predicted trend because the accuracy between the predicted
  value for my model give better results than the accuracy in predicted trend.

Conclusion
- This is a hard challenge for me to decide which is the best datasets trend among the 3 top. In here I can
  say top 1 is the best because I assume that my predict model outperform the predicted value and
  it shows top 1 gives the best result even though in predictd value gives top 3 is the best. So I can say
  that top 1 is best datasets, top 2 is underfit dataset and top 3 is overfit dataset.

p.s Please correct me if my analysis has mistakes, I have done my best on analyze this trends for 6 days straight

:3

Thank you.
