from sklearn.svm import SVC
from pathlib import Path 
import sys 
sys.path.append(str(Path(__file__).parent.parent))
from SupportVectorMachine.syntheticDataset import *
from irisDataSet import *
from plot_decision_regions import plot_decision_regions


svm = SVC(kernel='rbf', random_state=1, gamma=0.1, C=1000.0)

# svm.fit(X_xor, y_xor)
# plot_decision_regions(X=X_xor, y=y_xor, classifier=svm)
# plt.legend(loc='upper left')
# plt.tight_layout()
# plt.show()

svm.fit(X_train_std, y_train)
plot_decision_regions(X=X_combined_std, y=y_combined, classifier=svm, test_idx=range(105, 150))
plt.legend(loc='upper left')
plt.xlabel('Length petal [standardized]')
plt.ylabel('Width petal [standardized]')
plt.tight_layout()
plt.show()