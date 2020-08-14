# -*- coding: utf-8 -*-
"""
Created on Thu Aug 13 14:32:53 2020

@author: luisx

Regresion lineal simple en Python
"""

# %% Importación de las librerias

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

# carga de los datos
dataset = pd.read_csv('../Salary_Data.csv')
# La variable independiente (X)
# La variable dependiente o la que se va a predecir (Y)
# Se localizaran los elementos (Filas y columnas) por posicion
# Quiero todas las filas y X columnas. Extraeremos unicamente los valores
X = dataset.iloc[:, :-1].values
# Ahora nos interesa la ultima columna
y = dataset.iloc[:, -1].values

# %% Dividir el dataset en conjunto de entrenamiento y testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3,
                                                    random_state=0)

# %% Escalado
# Como la fila de salario es muy grande comparado con la edad, podrian
# haber problemas al calcular las distancias euclideas. De esa forma las
# variables pequeñas NO pasaran inadvertidas. Se escalarán los datos.
"""
Un algoritmo de regresion lineal simple no requiere escalado
from sklearn. preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
# Para que tenga el mismo escalado se usa esto.
X_test = sc_X.transform(X_test)

"""

# %% Crear modelo de Regresion Lineal Simple con el set de entrenamiento
from sklearn.linear_model import LinearRegression
regression = LinearRegression()
regression.fit(X_train, y_train)

# %% Predecir el conjunto test
y_pred = regression.predict(X_test)

# %% Visualizacion de los resultados del conjunto de entrenamiento
plt.scatter(X_train, y_train, color="red")
plt.plot(X_train, regression.predict(X_train), color="blue")
plt.title("Sueldo vs Experiencia (años). Conjunto de Entrenamiento")
plt.xlabel("Años de Experiencia")
plt.ylabel("Sueldo")
plt.show()

# %% Visualizacion de los resultados del conjunto de test
plt.scatter(X_test, y_test, color="red")
plt.plot(X_test, y_pred, color="blue") # Sera igual que la anterior.
plt.title("Sueldo vs Experiencia (años). Conjunto de Test")
plt.xlabel("Años de Experiencia")
plt.ylabel("Sueldo")
plt.show()
