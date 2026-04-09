import os
import pandas as pd
import numpy as np


s = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'

df = pd.read_csv(s, 
                 header=None,
                 encoding='utf-8'
                 )

# select setosa and versicolor
y = df.iloc[0:100, 4].values 
y = np.where(y == 'Iris-setosa', 0, 1)

X = df.iloc[0:100, [0, 2]].values

# print(df.tail())  
#        0    1    2    3               4
# 145  6.7  3.0  5.2  2.3  Iris-virginica
# 146  6.3  2.5  5.0  1.9  Iris-virginica
# 147  6.5  3.0  5.2  2.0  Iris-virginica
# 148  6.2  3.4  5.4  2.3  Iris-virginica
# 149  5.9  3.0  5.1  1.8  Iris-virginica

# улучшим градиентный спуск с помощью масштабирования признаков
# standardization
# (x_j)' = (x_j - m_j) / o_j, где x_j - вектор из значений j-того признака всех обучающих образцов n
# o_j - стандартное отклонение 
# m_j - среднее значение выборки 

X_std = np.copy(X)
X_std[:, 0] = (X[:, 0] - X[:, 0].mean()) / X[:, 0].std()
X_std[:, 1] = (X[:, 1] - X[:, 1].mean()) / X[:, 1].std()
