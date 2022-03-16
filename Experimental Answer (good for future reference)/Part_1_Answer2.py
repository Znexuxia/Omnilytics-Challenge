# Import any important package to the file
import pandas as pd
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import matplotlib.pyplot as plt

# Retrieve the data from the file
data_trend = "trends1_cleaned.csv"
dataset_trend = pd.read_csv(data_trend,parse_dates=['date'],index_col='date',dayfirst=(True))

data_predict = "predictions_cleaned.csv"
dataset_predict = pd.read_csv(data_predict,parse_dates=['date'],index_col='date',dayfirst=(True))

# Make an mean for each month value
data_month_trend = dataset_trend.resample('M').mean().astype("int")
data_month_predict = dataset_predict.resample('M').mean().astype("int")

# Make a class for calculation on last 12 months
class last_12M:
    def __init__(self,x,y):      
        # Visualize the data
        plt.plot(data_month_trend)
        plt.title('Time Series Data')
        plt.ylabel('Popularity of top')
        plt.xticks(rotation=45)
        plt.show()

        
        # Split-out validation dataset
        validation_size = 0.20
        seed = 7
        
        X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(x,y, test_size=validation_size, random_state = seed, shuffle=(True))
        
        # Test Harness
        # Test options and evaluation metric
        seed = 7
        scoring = 'accuracy'



        # Build Models
        # Spot check algorithms
        models = []
        models.append(('LR',LogisticRegression(solver='liblinear', multi_class='ovr')))
        models.append(('LDA', LinearDiscriminantAnalysis()))
        models.append(('KNN', KNeighborsClassifier()))
        models.append(('CART', DecisionTreeClassifier()))
        models.append(('NB', GaussianNB()))
        models.append(('SVM',SVC(gamma='auto')))
        
        # evaluate each model in turn
        results = []
        names = []
        for name, model in models:
            kfold = model_selection.KFold(n_splits=6, random_state = seed, shuffle=(True))
            cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold , scoring = scoring)
            results.append(cv_results)
            names.append(name)
            msg = "%s: %f (%f)" %(name, cv_results.mean(), cv_results.std())
            print(msg)
        
        # Select Best Model
        # Compare Algorithms
        fig = plt.figure()
        fig.suptitle('Algorith Comparison')
        ax = fig.add_subplot(111)
        plt.boxplot(results)
        ax.set_xticklabels(names)
        plt.show()


        # Make Predictions
        # Make predictions on validation dataset
        # knn = KNeighborsClassifier()
        # knn.fit(X_train, Y_train)
        # predictions = knn.predict(X_validation)
        # print(accuracy_score(Y_validation, predictions))
        # print(confusion_matrix(Y_validation, predictions))
        # print(classification_report(Y_validation, predictions))
        
        svm = SVC()
        svm.fit(X_train, Y_train)
        predictions = svm.predict(X_validation)
        print(accuracy_score(Y_validation, predictions))
        print(confusion_matrix(Y_validation, predictions))
        print(classification_report(Y_validation, predictions))
        
        print(Y_validation - predictions)
    
# Make a class for performance on last 12 months
class perf_12M:
    # Get a last 12 months dataset
    data_last_12m_trend = dataset_trend.last("12M").astype("int")
    data_last_12m_predict = dataset_predict.last("12M").astype("int")
    
    # Summarize the Dataset Trend
    # shape
    print(dataset_trend.shape)

    # Peek at the data
    # head
    print (dataset_trend.head(20))

    # Statistical Summary
    # descriptions
    print(dataset_trend.describe())
    
    
    # Visualize the data
    # plt.plot(data_last_12m)
    # plt.title('Trend of variables last 12 months')
    # plt.ylabel('Quantity of the data')
    # plt.xticks(rotation=45)
    # plt.show()
    
    
    # Separate each top for y output
    array_trend = data_last_12m_trend.values
    y_top1 = array_trend[:,0]
    y_top2 = array_trend[:,1]
    y_top3 = array_trend[:,2]
    
    print(len(y_top1))
    
    y_array = [y_top1, y_top2, y_top3]
    
    # Separate each top for x intput
    array_predict = data_last_12m_predict.values
    x_top1 = array_predict[:,0:3]
    x_top2 = array_predict[:,3:6]
    x_top3 = array_predict[:,6:9]
    
    print(len(x_top1))
    
    x_array = [x_top1, x_top2, x_top3]
    
    result = last_12M(x_top1, y_top1)
    
    # for x, y in zip(x_array , y_array):
        # result = last_12M(x,y)