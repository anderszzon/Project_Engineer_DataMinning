#LIBRERIAS
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import sklearn.metrics as metrics

# PREPROCESAMIENTO DEL DATASET
dataset = pd.read_csv('cultivos.csv')
dataset.head()
dataset.info()
dataset.describe()
dataset = pd.get_dummies(dataset, columns=['cultivo','municipio','PERIODO'] )
print(dataset.shape)

dataset.hist(bins = 40)

sns.set(style='whitegrid', context='notebook')
cols = ['area sembrada','area cosechada','produccion', 'rendimiento']
sns.pairplot(dataset[cols], height=2.5)
plt.show()

cm = np.corrcoef(dataset[cols].values.T)
sns.set(font_scale=1.5)
sns.heatmap(cm, cbar=True, annot=True,yticklabels=cols,xticklabels=cols)
plt.show()


# CREACION DEL MODELO REGRESION LINEAL
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

model = LinearRegression()
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

print("mse: ", mse.round(4))
print("mae: ", mae.round(4))
print("r2 ", r2.round(4))

dataset['mse'] = mse
dataset['mae'] = mae
dataset['r2'] = r2
dataset['rmse'] = rmse
dataset['prediction'] = model.predict(X)

dataset.to_csv('Prediccion.csv')

print('Fin del programa')

