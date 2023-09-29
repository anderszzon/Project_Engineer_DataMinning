import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn import preprocessing

dataset = pd.read_csv('HR_file.csv')

# CAMBIA LOS VALORES DE TEXTO A NUMERO
le = LabelEncoder()
dataset['Departments'] = le.fit_transform(dataset['Departments'])
dataset['salary'] = le.fit_transform(dataset['salary'])

y=dataset['Quit the Company']
features = ['Satisfaction Level','Last Evaluation','Number of Projects',
            'Monthly Hours','Total Time at the Company','Work Accidents',
            'Quit the Company','Promoted in Last 5 yrs','Departments','salary','Management']
x=dataset[features]
s=StandardScaler()
x=s.fit_transform(x)

x_train,x_test,y_train, y_Test = train_test_split(x,y)

log = LogisticRegression()
log.fit(x_train, y_train)
y_pred = log.predict(x)
y_prob = log.predict_proba(x)[:,1]

dataset['predictions'] = y_pred
dataset['probabilities'] = y_prob

print('Fin del programa')

