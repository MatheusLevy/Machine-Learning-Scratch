import numpy as np

class Linear_Regression():
    def __init__(self, X, y, alpha, num_iter):
        self.theta = np.zeros((X.shape[1] + 1, 1)) # theta is a vector of parameters for each feature plus one for the bias
        self.alpha = alpha
        self.num_iter = num_iter
        self.X = X
        self.y = y

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def h_x(self, X):
        ones = np.ones((X.shape[0], 1))
        concat = np.concatenate([ones, X], axis=1)
        return self.sigmoid(np.dot(concat, self.theta))

    def loss(self, X, y):
        m = X.shape[0] # num of Samples
        h = self.h_x(X) # prediction for each sample as a vecto
        return np.dot(y.T, np.log10(h)) + np.dot((1 - y).T, np.log10(1 - h)) # (maximize) Log Likehood function 

    def gradient_descent(self):
        m = self.X.shape[0]
        for _ in range(self.num_iter):
            error = self.y - self.h_x(self.X)
            # Att weights 
            self.theta[0] = self.theta[0] + self.alpha * np.sum(error)
            self.theta[1:] = self.theta[1:] + (self.alpha * np.dot(error.T, self.X)).T
            print(self.loss(self.X, self.y))

    def newton_method(self):
        m = self.X.shape[0]
        for _ in range(self.num_iter):
            error = self.y - self.h_x(self.X)
            ones = np.ones((self.X.shape[0], 1))
            X = np.concatenate([ones, self.X], axis=1)
            gradient = np.dot(error.T, X).T
            hessian = np.dot(X.T, (self.h_x(self.X) * (1 - self.h_x(self.X))) * X)
            self.theta = self.theta + np.dot(np.linalg.inv(hessian), gradient)
            print(self.loss(self.X, self.y))

        
X = np.array([[1,2], [1,1],   [3,0.2], [1,5], [2,4], [7,2],  [1,0.5], [2,0.1], [3,0.01], [3,3]])
y=  np.array([[0],    [0],     [0],     [1],   [1],   [1],    [0],     [0],       [0],    [1]])

lr = Linear_Regression(X, y, 0.01, 3)
lr.newton_method()
print(lr.h_x(np.array([[1,2], [1,1],   [3,0.2], [1,5], [2,4], [7,2],  [1,0.5], [2,0.1], [3,0.01], [3,3]])))
