import matplotlib.pyplot as plt 
import numpy as np 

# logical xor syntheticDataset
np.random.seed(1)
X_xor = np.random.randn(200, 2)
y_xor = np.logical_xor(X_xor[:, 0] > 0, 
                       X_xor[:, 1] > 0)

y_xor = np.where(y_xor, 1, 0)

# plt.scatter(X_xor[y_xor == 1, 0], 
#             X_xor[y_xor == 1, 1],
#             c='blue', marker='s', label='class1')

# plt.scatter(X_xor[y_xor == 0, 0], 
#             X_xor[y_xor == 0, 1],
#             c='red', marker='o', label='class0')
# plt.xlim([-3, 3])
# plt.ylim([-3, 3])
# plt.xlabel('feature1')
# plt.ylabel('feature2')
# plt.legend(loc='best')
# plt.tight_layout()
# plt.show()