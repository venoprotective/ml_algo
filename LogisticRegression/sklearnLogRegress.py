from sklearn.linear_model import LogisticRegression 
import sys 
from pathlib import Path 
sys.path.append(str(Path(__file__).parent.parent)) 
from IrisDataSet import *
from plot_decision_regions import plot_decision_regions
import matplotlib.pyplot as plt 

 
lr = LogisticRegression(C=100.0, solver='lbfgs', multi_class='ovr')

lr.fit(X_train_std, y_train)

plot_decision_regions(X=X_combined_std, y=y_combined,
                      classifier=lr, test_idx=range(105, 150)
                      )
plt.xlabel('Length petal [standardized]')
plt.ylabel('Width petal [standardized]')
plt.legend(loc='upper left')
plt.tight_layout()   
plt.show()