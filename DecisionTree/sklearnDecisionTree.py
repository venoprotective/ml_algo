from sklearn.tree import DecisionTreeClassifier 
from pathlib import Path
import sys 
sys.path.append(str(Path(__file__).parent.parent))
from irisDataSet import *
from plot_decision_regions import plot_decision_regions
import matplotlib.pyplot as plt 

 
tree_model = DecisionTreeClassifier(criterion='gini', max_depth=4, random_state=1)
tree_model.fit(X_train, y_train)

X_combined = np.vstack((X_train, X_test))
y_combined = np.hstack((y_train, y_test))

plot_decision_regions(X_combined, y_combined, tree_model, test_idx=range(105,150))
plt.xlabel('length petal')
plt.ylabel('width petal')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()