import pandas as pd

df = pd.read_excel('ICETEX 2.xlsx')

df.head(5)

df.columns

df.drop(columns=['LINEA DE CREDITO'], inplace=True)

df.isnull().sum(axis=0)

df.shape

df.dropna(axis=0, inplace=True)

df.shape


df

df['PROGRAMA'].value_counts()

#En la columna Programa reemplazamos algunos valores mal escritos por los valores correctos
df['PROGRAMA']=df['PROGRAMA'].str.replace('DERECCHO','DERECHO')
df['PROGRAMA']=df['PROGRAMA'].str.replace('PSICOLOGÍA','PSICOLOGIA')
df['PROGRAMA']=df['PROGRAMA'].str.replace('psicologia','PSICOLOGIA')
df['PROGRAMA']=df['PROGRAMA'].str.replace('CONTADURÍA PÚBLICA','CONTADURIA PUBLICA')
df['PROGRAMA']=df['PROGRAMA'].str.replace('COMUNICACIÓN SOCIAL','COMUNICACION SOCIAL')
df['PROGRAMA']=df['PROGRAMA'].str.replace('ADMINISTRACION DE EMPRESA','ADMINISTRACION DE EMPRESAS')
df['PROGRAMA']=df['PROGRAMA'].str.replace('ADMINISTRACIÓN DE NEGOCIO INTERNACIONALES','ADMINISTRACION DE NEGOCIOS INTERNACIONALES')



df['PROGRAMA'].value_counts()

df['PROGRAMA']=df['PROGRAMA'].str.replace('ADMINISTRACION DE EMPRESASS','ADMINISTRACION DE EMPRESAS')

df['PROGRAMA'].unique()

from sklearn.preprocessing import OrdinalEncoder
ord_enc = OrdinalEncoder()
df["PROGRAMA"] = ord_enc.fit_transform(df[["PROGRAMA"]])
df["APRUEBA"] = ord_enc.fit_transform(df[["APRUEBA"]])

df.head(5)

df['PROGRAMA']=df['PROGRAMA'].astype('int64')
df['SEMESTRE']=df['SEMESTRE'].astype('int64')
df['PROMEDIO']=df['PROMEDIO'].astype('float64')
df['APRUEBA']=df['APRUEBA'].astype('int64')

df.dtypes

df

import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

cols = ['PROGRAMA','SEMESTRE','PROMEDIO','APRUEBA']

cm = np.corrcoef(df[cols].values.T)
sns.set(font_scale=1.5)
hm = sns.heatmap(cm, cbar=True, 
                annot=True,
                square=True,
                fmt='.2f',
                annot_kws={'size':15},
                yticklabels=cols,
                xticklabels=cols                
                )
plt.show()

X = df.iloc[:, [1,2]].values
y = df.iloc[:, 3].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 27, shuffle = True)

from sklearn.tree import DecisionTreeClassifier
arbol = DecisionTreeClassifier()
arbol.fit(X_train, y_train)

y_pred = arbol.predict(X_test)

from sklearn.metrics import accuracy_score
puntaje = round(accuracy_score(y_test, y_pred), 3)
print('La exactitud del modelo es: {}'.format(puntaje))

#SEMESTRE 	PROMEDIO
datos_prueba = [[3,4]]
res = arbol.predict(datos_prueba)
r = res[0]
if r==1:
   print("Aprueba")
else:
   print("No Aprueba")

import joblib

from sklearn import linear_model

from sklearn.linear_model import LogisticRegression


lr = LogisticRegression()
lr.fit(X_train, y_train)

predicciones = lr.predict(X_test)
from sklearn.metrics import confusion_matrix, accuracy_score
confusion_matrix(predicciones, y_test)
print("La exactitud es: ", round(accuracy_score(predicciones, y_test),2))

joblib.dump(lr, 'modelo.pkl')



