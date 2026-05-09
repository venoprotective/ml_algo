import os
import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


s = 'https://gist.githubusercontent.com/curran/a08a1080b88344b0c8a7/raw/0e7a9b0a5d22642a06d3d5b9bcbad9890c8ee534/iris.csv'

df = pd.read_csv(s, 
                 header=None,
                 encoding='utf-8'
                 )
y_perceptron = df.iloc[1:, 4].values 
y_perceptron = np.where(y_perceptron == 'setosa', 0, 1)

X_perceptron = df.iloc[1:, [0, 2]].values
X_perceptron = X_perceptron.astype(np.float64)

# print(X[0], len(X))  
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
X_std_adaline = np.copy(X_perceptron)
X_std_adaline[:, 0] = (X_perceptron[:, 0] - X_perceptron[:, 0].mean()) / X_perceptron[:, 0].std()
X_std_adaline[:, 1] = (X_perceptron[:, 1] - X_perceptron[:, 1].mean()) / X_perceptron[:, 1].std()

iris = datasets.load_iris()
X = iris.data[:, [2,3]]
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)

sc = StandardScaler()
sc.fit(X_train)

X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

X_train_01_subset = X_train_std[(y_train == 0) | (y_train == 1)]
y_train_01_subset = y_train[(y_train == 0) | (y_train == 1)]

X_test_01_subset = X_test_std[(y_test == 0) | (y_test == 1)]
y_test_01_subset = y_test[(y_test == 0) | (y_test == 1)]
