from LogisticRegressionClass import LogisticRegressionGD 
import sys
from pathlib import Path  
sys.path.append(str(Path(__file__).parent.parent))
from plot_decision_regions import *
from irisDataSet import *
import matplotlib.pyplot as plt 


lrgd = LogisticRegressionGD(eta=0.3, n_iter=1000, random_state=1)
lrgd.fit(X_train_01_subset, y_train_01_subset)

plot_decision_regions(X=X_train_01_subset, y=y_train_01_subset, classifier=lrgd)
plt.xlabel("длина чашел стандартизирована")
plt.ylabel("длина лепестка стандартизирована")

plt.legend(loc = 'upper left')
plt.tight_layout()
plt.show()

# print(lrgd.accuracy(X=X_test_01_subset, y_true=y_test_01_subset))
# 1.0