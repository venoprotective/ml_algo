import matplotlib.pyplot as plt

from perceptron import Perceptron
from AdaptiveLinearNeuron import AdalineGD
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

# plot_decision_regions(X, y, classifier=ppn)
# plt.xlabel("длина чашел.")
# plt.ylabel("длина лепестка")
# plt.legend(loc='upper left')
# plt.show()


# test adaline with gradient descent 
# comparison of learning speed

# ada0 = AdalineGD(n_iter=15, eta=0.1)
# ada0.fit(X, y)

# ada1 = AdalineGD(n_iter=15, eta=0.0001)
# ada1.fit(X, y)

# fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10,4))
# ax[0].plot(range(1, len(ada0.losses_) + 1), np.log10(ada0.losses_) ,marker='o')
# ax[0].set_xlabel('Epochs')
# ax[0].set_ylabel('log mse')
# ax[0].set_title('adaline eta=0.1')
# ax[1].plot(range(1, len(ada1.losses_) + 1), ada1.losses_ ,marker='o')
# ax[1].set_xlabel('Epochs')
# ax[1].set_ylabel('mse')
# ax[1].set_title('adaline eta=0.0001')
# plt.show()


# testing after standardization 
ada = AdalineGD(n_iter=20, eta=0.5)
ada.fit(X_std, y)
plot_decision_regions(X_std, y, classifier=ada)
plt.title('Adaline with GD')
plt.xlabel('длина чашел.')
plt.ylabel('длина лепестка')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()
print('113212iprjdpflskfnsrpenkpdsfmepfkfknwigoeznfleki4')
# measurement MSE 
plt.plot(range(1, len(ada.losses_) + 1), ada.losses_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('MSE')
plt.tight_layout()
plt.show()