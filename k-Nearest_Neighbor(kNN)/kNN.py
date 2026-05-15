import numpy as np


class kNN:
    def __init__(self, k=5):
        self.k = k 
        self.X_train = None
        self.y_train = None 
    
    def fit(self, X, y):
        self.X_train = np.array(X)
        self.y_train = np.array(y)
        
        return self 
    
    def _calculate_distance(self, X_test):
        '''векторизованный расчёт евклидова расстояния'''
        test_sq = np.sum(X_test ** 2, axis=1).reshape(-1, 1)
        train_sq = np.sum(self.X_train ** 2, axis=1).reshape(1, -1)
        
        X_test = np.array(X_test)
        
        distance = test_sq + train_sq - 2 * np.dot(X_test, self.X_train.T) 
        
        return distance 
    
    def predict(self, X):
        distances = self._calculate_distance(X)
        
        k_nearest_indices = np.argsort(distances, axis=1)[:, :self.k]
        k_nearest_labels = self.y_train[k_nearest_indices]    
        
        predictions = []
        for label in k_nearest_labels:
            unique, count = np.unique(label, return_counts=True)
            predictions.append(unique[np.argmax(count)])        
            
        return np.array(predictions)