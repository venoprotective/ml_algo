from sklearn.linear_model import LogisticRegression
import sys
from pathlib import Path  
sys.path.append(str(Path(__file__).parent.parent))
from plot_decision_regions import *
from IrisDataSet import *
import matplotlib.pyplot as plt 


weights, params = [], []
for c in np.arange(-7, 7):
    lr = LogisticRegression(C=10.0 ** c, multi_class='ovr')
    lr.fit(X_train_std, y_train)
    weights.append(lr.coef_[1])
    params.append(10.0 ** c)
weights = np.array(weights)
plt.plot(params, weights[:, 0], label='length petal')
plt.plot(params, weights[:, 1], linestyle='--', label='width petal')
plt.xlabel('C')
plt.xscale('log')
plt.ylabel('weight coef')
plt.axvline(x=100, color='red', linewidth=1, linestyle='--', label='optimal coef')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()

