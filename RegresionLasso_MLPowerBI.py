# 'dataset' holds the input data for this script
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import  LinearRegression, Lasso

dataset = pd.read_csv('diamonds2.csv')
lasso = Lasso()
X = dataset.drop(['price','x','y','z'],axis=1)
X = pd.get_dummies(X)
y=dataset['price']
features = X.columns
s = StandardScaler()
X = s.fit_transform(X)
X_train,X_test,y_train,y_test = train_test_split(X,y)
lasso.fit(X_train,y_train)

y_pred = lasso.predict(X_test)

RMSE = np.sqrt(mean_squared_error(y_test,y_pred))
dataset['prediction'] = lasso.predict(X)
dataset['ERROR CUADRATICO MEDIO'] = RMSE

print('Fin del programa')
