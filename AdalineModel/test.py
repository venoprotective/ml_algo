from AdaptiveLinearNeuron import AdalineGD 
import sys
from pathlib import Path  
sys.path.append(str(Path(__file__).parent.parent))
from plot_decision_regions import *
from irisDataSet import *
import matplotlib.pyplot as plt 


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
ada.fit(X_std_adaline, y_perceptron)
plot_decision_regions(X_std_adaline, y_perceptron, classifier=ada)
plt.title('Adaline with GD')
plt.xlabel('длина чашел.')
plt.ylabel('длина лепестка')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()
# measurement MSE 
plt.plot(range(1, len(ada.losses_) + 1), ada.losses_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('MSE')
plt.tight_layout()
plt.show()