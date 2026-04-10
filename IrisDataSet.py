import os
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris

s = 'https://gist.githubusercontent.com/curran/a08a1080b88344b0c8a7/raw/0e7a9b0a5d22642a06d3d5b9bcbad9890c8ee534/iris.csv'

# df = pd.read_csv(s, 
#                  header=None,
#                  encoding='utf-8'
#                  )
iris = load_iris()
# select setosa and versicolor
# y = df.iloc[1:, 4].values 
# y = np.where(y == 'setosa', 0, 1)

# X = df.iloc[1:, [0, 2]].values

y = iris.target[:100]
y = np.where(y==0, 0, 1)
X = iris.data[:100, [0, 2]]


print(X[0], len(X))  
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
X = X.astype(np.float64)

# X_std = np.copy(X)
# X_std[:, 0] = (X[:, 0] - X[:, 0].mean()) / X[:, 0].std()
# X_std[:, 1] = (X[:, 1] - X[:, 1].mean()) / X[:, 1].std()
X_std = ((X - X.mean(axis=0)) / X.std(axis=0))