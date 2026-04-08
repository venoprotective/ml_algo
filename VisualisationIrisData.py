import matplotlib.pyplot as plt
import numpy as np

from IrisDataSet import *


# зависимость длины лепестка от длины чашелистика
plt.scatter(X[:50, 0], X[:50, 1], color='red', marker='o', label='Setosa')
plt.scatter(X[50:100, 0], X[50:100, 1], color='blue', marker='^', label='Versicolor')
plt.xlabel("Длина чашел.")
plt.ylabel("Длина лепестка")
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()
