# 'dataset' holds the input data for this script
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import  LinearRegression, Lasso
from sklearn.ensemble import RandomForestRegressor


dataset = pd.read_csv('diamonds2.csv')
#rf = RandomForestRegressor(n_estimators=10)
rf = RandomForestRegressor()
X = dataset.drop(['price','x','y','z'],axis=1)
X = pd.get_dummies(X)
y=dataset['price']
features = X.columns
s = StandardScaler()
X = s.fit_transform(X)
X_train,X_test,y_train,y_test = train_test_split(X,y)
rf.fit(X_train,y_train)

y_pred = rf.predict(X_test)
#y_pred2=y_pred = rf.predict(X_test)


RMSE = np.sqrt(mean_squared_error(y_test,y_pred))
dataset['prediction'] = rf.predict(X)
dataset['ERROR CUADRATICO MEDIO'] = RMSE

print('Fin del programa')
