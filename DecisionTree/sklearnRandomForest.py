from sklearn.ensemble import RandomForestClassifier 
from pathlib import Path
import sys 
sys.path.append(str(Path(__file__).parent.parent))
from irisDataSet import *
from plot_decision_regions import plot_decision_regions
import matplotlib.pyplot as plt 


forest = RandomForestClassifier(n_estimators=25, 
                                random_state=1, 
                                max_depth=4,
                                n_jobs=1)
forest.fit(X_train, y_train)
plot_decision_regions(X_combined, y_combined, classifier=forest, test_idx=range(105, 150))
plt.xlabel('length petal')
plt.ylabel('width petal')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()