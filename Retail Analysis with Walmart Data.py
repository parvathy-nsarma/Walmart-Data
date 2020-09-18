
# Retail Analysis with Walmart Data

#Import pandas library
import pandas as pd
import numpy as np

#Import the csv file where details of Walmart data is contained
df_wss = pd.read_csv('Walmart_Store_sales.csv')
df_wss.head()


#View the shape of dataframe

df_wss.shape


#Checking for null values
df_wss.isnull().sum()


#Decribe dataframe
df_wss.describe()


# Basic Statistic Task

# Which store has maximum sales

#Values of the Total Weekly_Sales for each store
df_groupby = df_wss.groupby('Store')['Weekly_Sales'].sum()
print(df_groupby.shape)
print(df_groupby.head())


#Store with Maximum Sales and Value of the Sales
print( 'Store Number {} has maximum total weekly sales of {}.'.format (df_groupby.idxmax(),df_groupby.max()))


# Which store has maximum standard deviation i.e., the sales vary a lot. Also,to find out the coefficient of mean to standard deviation

#Value of Standard Deviations
df_groupstd = df_wss.groupby('Store')['Weekly_Sales'].std()
print(df_groupstd.shape)
print(df_groupstd)


#Store with Maximum Standard Deviation
print( 'Store Number {} has Maximum Standard Deviation of {}.'.format (df_groupstd.idxmax(),df_groupstd.max()))


#Coefficient of mean to standard deviation
df_groupmean = df_wss.groupby('Store')['Weekly_Sales'].mean()
print(df_groupmean.shape)
cms = (df_groupstd.sum()/df_groupmean.sum())*100
print(cms)


# Which store/s has good quarterly growth rate in Q3’2012

#Store/s that has good quarterly growth rate in Q3’2012(July 1 to September 30)

df_Q32012=df_wss[(pd.to_datetime(df_wss['Date'])>= pd.to_datetime('01-07-2012'))&(pd.to_datetime(df_wss['Date'])<= pd.to_datetime('30-09-2012'))]

df_wss_growth = df_Q32012.groupby(['Store'])['Weekly_Sales'].sum()

print("Store Number {} has a good Quartely Growth in 3rd Quarter(Q3) of 2012 {}".format(df_wss_growth.idxmax(),df_wss_growth.max()))


# Some holidays have a negative impact on sales.
# Find out holidays which have higher sales than the mean sales in non-holiday season for all stores together

# Stores Holiday
stores_holiday_sales = df_wss[df_wss['Holiday_Flag'] == 0]
#Store_non_holiday
stores_nonholiday_sales = df_wss[df_wss['Holiday_Flag'] == 1]


stores_holiday_sales_mean = df_wss[(df_wss['Holiday_Flag'] == 0)]['Weekly_Sales'].mean()
stores_nonholiday_sales_sum = df_wss[(df_wss['Holiday_Flag'] == 1)].groupby('Date')['Weekly_Sales'].sum()

print(stores_nonholiday_sales_sum>stores_holiday_sales_mean)


#  Correlation Plot using heatmap

#import seaborn and matplotlib library
import matplotlib.pyplot as plt
import seaborn as sns

#To view plot in Jupyter

get_ipython().run_line_magic('matplotlib', 'inline')

corr = df_wss.corr()
plt.figure(figsize=(10,10))
sns.heatmap(corr, annot=True)             
plt.plot()


# Correlation Values

df_wss[['Store','CPI','Fuel_Price','Unemployment','Weekly_Sales']].corr()   


# Change dates into days by creating new variable.

df_wss['Days'] = pd.to_datetime(df_wss['Date']).dt.day_name()

df_wss.head()


# Statistical Model
# 
# For Store 1 – Build  prediction models to forecast demand
# 
# Linear Regression 

# Utilize variables like date and restructure dates as 1 for 5 Feb 2010 (starting from the earliest date in order).
# 

# SLR

#import LinearRegression model from sklearn package 
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn import metrics
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score


#Feature
X_features = df_wss[df_wss['Store'] ==1][['Store','Date']]
next_date = df_wss[df_wss['Store'] ==1]['Date']
next_date.index +=1
X_features.Date = next_date.index
print(X_features.shape)
print(X_features.head())


#Target
y_targets = df_wss[df_wss['Store']==1][['Store','Weekly_Sales']] #Store 1 Weekly_Sales is the target

print(y_targets.shape)
print(y_targets.head())


#Train and Test split
X_train,X_test,y_train,y_test = train_test_split(X_features,y_targets,random_state = 21)


#Calling Linear Regression and fitting the model
lm = LinearRegression()
lm.fit(X_train,y_train)


#Intercept and Coefficient Values
print("Intercept: ",lm.intercept_)
print("Coefficient: ",lm.coef_)


#Predicting using the feature test values 
y_pred_slr = lm.predict(X_test)


#Root mean square error value and score(Accuracy) calculation
print("RMSE Value: ",np.sqrt(metrics.mean_squared_error(y_pred_slr,y_test))) 
accuracy = metrics.r2_score(y_test,y_pred_slr)
print("Accuracy: ",accuracy)


#Scatter Plot for predicted values
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

plt.scatter(y_test,y_pred_slr) 
plt.show()


# MLR

#Feature
X_feature = df_wss[df_wss['Store'] ==1][['Store','CPI', 'Unemployment','Fuel_Price']]
print(X_feature.head())
#Target
y_target = df_wss[df_wss['Store']==1][['Store','Weekly_Sales']] #Store 1 Weekly_Sales is the target
print(y_target.head())


#Train and Test split
X_train,X_test,y_train,y_test = train_test_split(X_feature,y_target,random_state = 21)


#Calling Linear Regression and fitting the model
linreg = LinearRegression()
linreg.fit(X_train,y_train)


#Predicting using the feature test values
y_pred_mlr = linreg.predict(X_test)


#Intercept and Coefficient Values
print("Intercept: ",linreg.intercept_)
print("Coefficient: ",linreg.coef_)


#Root mean square error value and r2_score(Accuracy) calculation
print("RMSE Value: ",np.sqrt(metrics.mean_squared_error(y_pred_mlr,y_test))) 
accuracy = metrics.r2_score(y_test,y_pred_mlr)
print("Accuracy: ",accuracy)


#Scatter Plot for predicted values
plt.scatter(y_test,y_pred_mlr) 
plt.show()


# Polynomial Regression

from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


#Transform features for polynomial regression 
X_feature_PolyReg = PolyReg.fit_transform(X_feature)
#Degree of polynomial is 3
PolyReg = PolynomialFeatures(degree = 3)
# Pipeline is created by creating a list of tuples including the name of model or estimator and its correspondign constructor
#for Polynomial regression
Pip_Input =[('scale',StandardScaler()), ('polynomial', PolynomialFeatures(include_bias=False)),('model',LinearRegression())]
# Pipeline the above input
pipe = Pipeline(Pip_Input)
print(pipe)


# Fit the model and predicting the first four 'Weekly_Sales' using Polynomial regression
pipe.fit(X_feature_PolyReg,y_target)

Ypipehat = pipe.predict(X_feature_PolyReg)
print(Ypipehat[0:4])


#shape of features in Polyreg
print(X_feature.shape)
print(X_feature_PolyReg.shape)


from math import sqrt
from sklearn. metrics import r2_score
from sklearn.metrics import mean_squared_error

#Root mean square error value and score(Accuracy) calculation
r_squared = r2_score(y_target, Ypipehat)
print('R-squared :', r_squared)
print('RMSE:' , sqrt(mean_squared_error(y_target,Ypipehat)))


# CPI positive impact on sales
# 

# Unemployment positive impact on sales

# Fuel_Price negative impact on sales

# Select the model which gives best accuracy.

# Polynomial Regression with degree 3 gives the best accuracy of 85%

# Simple Linear Regression gives 48% of accuracy and
# Multiple Linear Regression gives 44% of accuracy





