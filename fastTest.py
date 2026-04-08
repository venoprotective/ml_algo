import matplotlib.pyplot as plt

from perceptron import Perceptron
from IrisDataSet import *
from plot_decision_regions import plot_decision_regions


ppn = Perceptron(eta=0.1, n_iter=10)
ppn.fit(X, y)

# ошибки от эпох
# plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker='o')
# plt.xlabel("Epochs")
# plt.ylabel("Amount errors")
# plt.grid(True)
# plt.tight_layout()
# plt.show()

# test plot decision regions func

plot_decision_regions(X, y, classifier=ppn)
plt.xlabel("длина чашел.")
plt.ylabel("длина лепестка")
plt.legend(loc='upper left')
plt.show()