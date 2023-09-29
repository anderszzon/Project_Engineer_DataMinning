import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import mean_squared_error
import seaborn as sns
import matplotlib.pyplot as plt

#load in the data
df = pd.read_csv('cultivos.csv')

#Check the head of the data
df.head()
df.info()
df.describe()
df.columns
df.dtypes
                                                               #df = df.drop('Unnamed: 0', axis =1)
df.head()
                                                               #sns.heatmap(df.corr(),annot = True);
#feature engineering
#df['symmetry'] = df['x']/df['y']
df.head()
df = df.dropna(axis=0)
df_trans=pd.get_dummies(df)
X = df_trans.drop(['rendimiento'],axis=1)
#X = df_trans.drop(['rendimiento','cultivo','municipio','PERIODO'],axis=1)
y=df_trans['rendimiento']
features = X.columns
plt.figure(figsize=(20,10))
sns.heatmap(df_trans.corr(),annot=True);

#scale the data
s = StandardScaler()
X = s.fit_transform(X)

X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=0.80, random_state=0)
models_eval = pd.DataFrame(index=['Null','KNN','MLR'],columns=['RMSE'])
ypred_null = y_train.mean()

rf = RandomForestRegressor(random_state=0, n_estimators=500)
rf.fit(X_train,y_train)
y_pred2 = rf.predict(X_test)
##################################################################################################################

#knn = KNeighborsRegressor(n_neighbors=7)
knn = KNeighborsRegressor()
knn.fit(X_train,y_train)
y_pred = knn.predict(X_test)

lin = LinearRegression()
lin = lin.fit(X_train,y_train)
y_pred3 = lin.predict(X_test)

lasso = Lasso()
lasso.fit(X_train,y_train)
y_pred4 = lasso.predict(X_test)

##################################################################################################################

model_eval=pd.DataFrame(index=['KNN','MLR'],columns=['RMSE'])
model_eval.loc['KNN','RMSE']=np.sqrt(mean_squared_error(y_test,y_pred))
model_eval.loc['RF','RMSE'] = np.sqrt(mean_squared_error(y_test,y_pred2))
model_eval.loc['MLR','RMSE'] = np.sqrt(mean_squared_error(y_test,y_pred3))
model_eval.loc['Lasso','RMSE'] = np.sqrt(mean_squared_error(y_test,y_pred4))
model_eval.loc['NULL','RMSE'] = ypred_null

##################################################################################################################

fig, ax = plt.subplots(figsize=(12,8))
ax.scatter(y_test,y_pred2,s=1)
ax.plot(y_test,y_test,color='red')
plt.show()

sns.displot(y_pred-y_test)

##################################################################################################################

datapred1 = pd.DataFrame(y_pred)
datapred2 = pd.DataFrame(y_pred2)
datapred3 = pd.DataFrame(y_pred3)
datapred4 = pd.DataFrame(y_pred4)

df['pred knn'] = datapred1
df['pred rf'] = datapred2
df['pred rl'] = datapred3
df['pred rlaso'] = datapred4

df.to_csv('Prediccion_Python.csv')

"""
lin=LinearRegression()
lin.fit(X_train,y_train)
y_pred2 = lin.predict(X_test)
model_eval.loc['MLR','RMSE']=np.sqrt(mean_squared_error(y_pred2,y_test))

#features importance
from sklearn.ensemble import ExtraTreesRegressor
model =ExtraTreesRegressor()
model.fit(X,y)

model.feature_importances_.tolist()
"""

print('Fin del programa')
