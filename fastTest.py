import matplotlib.pyplot as plt

from perceptron import Perceptron
from IrisDataSet import *

ppn = Perceptron(eta=0.1, n_iter=10)
ppn.fit(X, y)

plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker='o')
plt.xlabel("Epochs")
plt.ylabel("Amount errors")
plt.grid(True)
plt.tight_layout()
plt.show()