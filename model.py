# Importing the libraries
import numpy as np
import pandas as pd
import pickle

dataset = pd.read_csv("E:\Chrome downloads\ML\ML1\Salary_Data.csv")

X = dataset.iloc[:,0]

y = dataset.iloc[:,1]

X = X.values.reshape(-1,1)
y = y.values.reshape(-1,1)

#Splitting Training and Test Set
#Since we have a very small dataset, we will train our model with all availabe data.
#from sklearn.model_selection import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()

#Fitting model with trainig data
regressor.fit(X, y)


y_pred=regressor.predict(X)

from sklearn import metrics
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y, y_pred)))

######################################################################################################

X1=np.log(X)
regressor1=LinearRegression()
regressor1.fit(X1,y)


y_pred1=regressor1.predict(X1)

from sklearn import metrics
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y, y_pred1)))

########################################################################################################


y1=np.log(y)
regressor2=LinearRegression()
regressor2.fit(X,y1)


y_pred2=regressor2.predict(X)
y_pred3=np.exp(y_pred2)

from sklearn import metrics
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y, y_pred3)))

########################################################################################################


X4=X*X
regressor3=LinearRegression()
regressor3.fit(X4,y)


y_pred4=regressor3.predict(X4)

from sklearn import metrics
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y, y_pred4)))

########################################################################################################


# Saving model to disk
pickle.dump(regressor, open('model_salary.pkl','wb'))


