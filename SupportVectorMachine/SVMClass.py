import numpy as np 

class SVMClass:
    def __init__(self, n_iter=1000, eta=0.01, C=100, random_state=1):
        self.n_iter = n_iter 
        self.eta = eta
        self.C = C 
        self.random_state = random_state 
        
    def fit(self, X, y):
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=X.shape[1])
        self.b_ = np.float64(0.0)
        self.losses_ = []
        
        y_ = np.where(y <= 0, -1, 1)
        
        for _ in range(self.n_iter):
            margins = y_ * self.net_input(X)
            violations = margins < 1
            
            # градиент по w 
            # регуляризация w / C (all)
            grad_w_reg = self.w_ / self.C
            # хинге лосс для нарушителей -y_i * x_i
            grad_w_loss = -np.dot(X[violations].T, y_[violations])  
            # собираем градиент
            grad_w = grad_w_reg + grad_w_loss / X.shape[0]
            
            grad_b = -np.sum(y_[violations]) / X.shape[0] if np.any(violations) else 0 
            
            # update weights 
            self.w_ -= self.eta * grad_w 
            self.b_ -= self.eta * grad_b 
            
            losses = self.hinge_loss(X, y_)
            loss = np.mean(losses) + np.dot(self.w_, self.w_) /(2 * self.C)
            self.losses_.append(loss)
        
        return self  
    
    def hinge_loss(self, X, y):
        return np.maximum(0, 1 - y * self.net_input(X)) # y_i * ((w, x_i) + b) >= 1, i=1,…,N
            
    def net_input(self, X):
        return np.dot(X, self.w_) + self.b_ 
    
    def predict(self, X):
        return np.where(self.net_input(X) >= 0, 1, 0)
    