from sklearn.linear_model import LogisticRegression 
import sys 
from pathlib import Path 
sys.path.append(str(Path(__file__).parent.parent)) 
from irisDataSet import *
from plot_decision_regions import plot_decision_regions
import matplotlib.pyplot as plt 
from sklearn.metrics import accuracy_score
 
lr_ovr = LogisticRegression(C=100.0, solver='lbfgs', multi_class='ovr')
lr_multinomial = LogisticRegression(C=100.0, solver='lbfgs', multi_class='multinomial')

lr_ovr.fit(X_train_std, y_train)
lr_multinomial.fit(X_train_std, y_train)

y_pred_ovr = lr_ovr.predict(X_test_std)
y_pred_multinomial = lr_multinomial.predict(X_test_std)

accuracy_score_ovr = accuracy_score(y_test, y_pred_ovr)
accuracy_score_multinomial = accuracy_score(y_test, y_pred_multinomial)

print(f'ovr accuracy {accuracy_score_ovr:.4f}')
print(f'multinomial accuracy {accuracy_score_multinomial:.4f}')
print(f'difference {abs(accuracy_score_multinomial - accuracy_score_ovr):.4f}')

# print(lr_ovr.predict_proba(X_test_std[:3, :]))

# plot_decision_regions(X=X_combined_std, y=y_combined,
#                       classifier=lr, test_idx=range(105, 150)
#                       )
# plt.xlabel('Length petal [standardized]')
# plt.ylabel('Width petal [standardized]')
# plt.legend(loc='upper left')
# plt.tight_layout()   
# plt.show()
