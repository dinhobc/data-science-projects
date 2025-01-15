# regressao_linear.py

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Carregar um dataset exemplo
url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/household_power_consumption.csv'
df = pd.read_csv(url, sep=";")

# Selecionando as colunas e tratando valores nulos
df = df[['Global_active_power']].dropna()

# Preparando os dados para o modelo
X = df.index.values.reshape(-1, 1)
y = df['Global_active_power'].values

# Dividir em dados de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Criação do modelo de regressão linear
model = LinearRegression()
model.fit(X_train, y_train)

# Fazer previsões
y_pred = model.predict(X_test)

# Plotar o gráfico
plt.plot(X_test, y_test, color='blue', label='Valor Real')
plt.plot(X_test, y_pred, color='red', label='Valor Predito')
plt.legend()
plt.title("Previsão de Consumo de Energia - Regressão Linear")
plt.show()