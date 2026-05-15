from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from pathlib import Path
import sys 
sys.path.append(str(Path(__file__).parent.parent))
from irisDataSet import *
from plot_decision_regions import plot_decision_regions
import matplotlib.pyplot as plt 


knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_std, y_train)

plot_decision_regions(X_combined_std, y_combined, classifier=knn, test_idx=range(105,150))
plt.xlabel('length petal')
plt.ylabel('width petal')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()
