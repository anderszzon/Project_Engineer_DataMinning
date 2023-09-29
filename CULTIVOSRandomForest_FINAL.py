#LIBRERIAS
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import sklearn.metrics as metrics
from sklearn.ensemble import RandomForestRegressor

dataset = pd.read_csv('cultivos.csv')
dataset.head()


# CREACION DEL MODELO RANDOMFOREST
#X_cols = list(set(dataset.columns)-set(['rendimiento']))

numerical = dataset.filter(["area cosechada","area sembrada","produccion"])
numerical.head()

categorical = dataset.filter(["cultivo","municipio","PERIODO"])
categorical.head()

cat_categorical = pd.get_dummies(categorical, drop_first=True)
cat_categorical.head()

X = pd.concat([numerical,cat_categorical], axis=1)
X.head()

Y = dataset.filter(["rendimiento"])

x = X.values
y = Y.values

X_train, X_test, y_train, y_test = train_test_split(x,y, test_size=0.20, random_state=0)

sc_x = StandardScaler()
sc_y = StandardScaler()
#sc = StandardScaler()

X_train = sc_x.fit_transform(X_train)
X_test = sc_y.fit_transform(X_test)
y_train = sc_y.fit_transform(y_train)
y_test = sc_y.fit_transform(y_test)

model=RandomForestRegressor(random_state=42, n_estimators=500)
regresor = model.fit(X_train, y_train.ravel())
#model.fit(X_train,y_train)
y_pred = model.predict(X_test)

print(y_pred.shape)

# EVALUACION DEL MODELO

r2 = metrics.r2_score(y_test, y_pred)
mse = metrics.mean_squared_error(y_test,y_pred)
mae = metrics.mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mse)

y_test = y_test.reshape(-1)
residuals = np.subtract(y_test,y_pred.reshape(-1))
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
