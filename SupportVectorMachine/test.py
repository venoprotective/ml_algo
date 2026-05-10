from SVMClass import SVMClass
from sklearn.svm import SVC 
import sys
from pathlib import Path  
sys.path.append(str(Path(__file__).parent.parent))
from IrisDataSet import *
from plot_decision_regions import plot_decision_regions
import matplotlib.pyplot as plt 


svm = SVC(kernel='linear', C=1.0, random_state=1)
my_svm = SVMClass(C=1.0, random_state=1)

svm.fit(X_train_std, y_train)
my_svm.fit(X_train_std, y_train)

plot_decision_regions(X_combined_std, y_combined, classifier=svm, test_idx = range(105, 150))
plt.xlabel("длина чашел.")
plt.ylabel("длина лепестка")
plt.legend(loc='upper left')
plt.show()

