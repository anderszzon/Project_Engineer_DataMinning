
#Load in our essentials
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import *
from sklearn.metrics import mean_squared_error

dataset = pd.read_csv('cultivos.csv')
dataset.head()

sns.set(style='whitegrid', context='notebook')
cols = ['area sembrada','area cosechada','produccion', 'rendimiento']
sns.pairplot(dataset[cols], height=2.5)

plt.show()

cm = np.corrcoef(dataset[cols].values.T)
sns.set(font_scale=1.5)
sns.heatmap(cm, cbar=True, annot=True,yticklabels=cols,xticklabels=cols)

plt.show()

X = dataset['produccion'].values.reshape(-1, 1)
y = dataset['rendimiento'].values.reshape(-1, 1)

sc_x = StandardScaler()
sc_y = StandardScaler()

X_std = sc_x.fit_transform(X)
y_std = sc_y.fit_transform(y)

slr = LinearRegression()
slr.fit(X_std, y_std)


plt.scatter(X_std,y_std)
plt.plot(X_std,slr.predict(X_std),color = 'red')
plt.ylabel("produccion [produccion]")
plt.xlabel("rendimiento [rendimiento]")

plt.show()

num_habitaciones = 5
num_habitaciones_std = sc_x.transform(np.array([num_habitaciones]).reshape(-1,1))
print("El precio de una casa con 5 habitaciones en Boston es de ", sc_y.inverse_transform(slr.predict(num_habitaciones_std)))


#Load the data and create the data variables
X = dataset.iloc[:,0:3]
y=dataset['rendimiento']
X_train,X_test,y_train,y_test =train_test_split(X,y)

# Create and fit the model for prediction
lin = LinearRegression()
lin.fit(X_train,y_train)
y_pred =lin.predict(X_test)

RMSE = np.sqrt(mean_squared_error(y_test,y_pred))



print('Fin del programa')