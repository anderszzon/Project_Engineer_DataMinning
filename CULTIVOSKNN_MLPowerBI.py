#LIBRERIAS
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import sklearn.metrics as metrics



dataset = pd.read_csv('cultivos.csv')
dataset = pd.get_dummies(dataset, columns=['cultivo','municipio','PERIODO'] )
dataset.head()

# CREACION DEL MODELO KNN
X_cols = list(set(dataset.columns)-set(['rendimiento']))
y_col = ['rendimiento']

X = dataset[X_cols].values
y = dataset[y_col].values

X_train, X_test, y_train, y_test = train_test_split(X,y)
sc_x = StandardScaler().fit(X)
sc_y = StandardScaler().fit(y)

X_train = sc_x.transform(X_train)
X_test = sc_x.transform(X_test)
y_train = sc_y.transform(y_train)
y_test = sc_y.transform(y_test)

model=KNeighborsRegressor()
model.fit(X_train,y_train)
y_pred = model.predict(X_test)

print(y_pred.shape)

# EVALUACION DEL MODELO

r2 = metrics.r2_score(y_test, y_pred)
mse = metrics.mean_squared_error(y_test,y_pred)
mae = metrics.mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mse)

residuals = np.subtract(y_test,y_pred)
plt.scatter(y_pred,residuals)
#plt.grid(color = "red")
plt.axhline(y=0, color='r', linestyle='-')
plt.show()

print("r2 ", r2.round(4))
print("mse: ", mse.round(4))
print("mae: ", mae.round(4))
print("rmse: ", rmse.round(4))

dataset['r2'] = r2
dataset['mse'] = mse
dataset['mae'] = mae
dataset['rmse'] = rmse
dataset['prediction'] = model.predict(X)

dataset.to_csv('Prediccion.csv')

print('Fin del programa')
