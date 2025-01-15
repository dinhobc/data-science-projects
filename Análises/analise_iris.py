# analise_iris.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Carregar o dataset
url = 'https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv'
df = pd.read_csv(url)

# Visualizar as primeiras linhas do dataset
print(df.head())

# Estatísticas descritivas
print(df.describe())

# Visualização de distribuições
sns.pairplot(df, hue="species")
plt.show()