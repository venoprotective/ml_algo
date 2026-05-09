from perceptron import Perceptron
import sys
from pathlib import Path  
sys.path.append(str(Path(__file__).parent.parent))
from IrisDataSet import *
from plot_decision_regions import plot_decision_regions
import matplotlib.pyplot as plt 


ppn = Perceptron(eta=0.1, n_iter=10)
ppn.fit(X_perceptron, y_perceptron)

# ошибки от эпох
# plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker='o')
# plt.xlabel("Epochs")
# plt.ylabel("Amount errors")
# plt.grid(True)
# plt.tight_layout()
# plt.show()

# test plot decision regions func

plot_decision_regions(X_perceptron, y_perceptron, classifier=ppn)
plt.xlabel("длина чашел.")
plt.ylabel("длина лепестка")
plt.legend(loc='upper left')
plt.show()
