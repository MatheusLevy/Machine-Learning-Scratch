import numpy as np

class SoftmaxRegression():
    def __init__(self, X, y, alpha, num_iter):
        self.num_classes = len(y[0])
        self.thetas_classes = np.zeros((X.shape[1] + 1, self.num_classes))
        self.alpha = alpha
        self.num_iter = num_iter
        self.X = X
        self.y = y

    def softmax(self, z):
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)
    
    def h_x(self, X):
        ones = np.ones((X.shape[0], 1))
        concat = np.concatenate([ones, X], axis=1)
        return self.softmax(np.dot(concat, self.thetas_classes))
    
    def Cross_Entropy(self, X, y):
        m = X.shape[0]
        h = self.h_x(X)
        return -np.sum(y * np.log(h))
    
    def gradient_descent(self):
        m = self.X.shape[0]
        for _ in range(self.num_iter):
            error = self.y - self.h_x(self.X)           

            self.thetas_classes[0] = self.thetas_classes[0] + self.alpha * np.sum(error, axis=0)
            self.thetas_classes[1:] = self.thetas_classes[1:] + (self.alpha * np.dot(self.X.T, error))
            print(self.Cross_Entropy(self.X, self.y))
            
    
X = np.array([ [1,1],     [3,0.02],  [1,0.4],  [1,3]])
y=  np.array([ [1,0,0],   [1,0,0],   [1,0,0], [0,1,0]])

SR = SoftmaxRegression(X, y, 0.001, 5000)
SR.gradient_descent()
predictions = SR.h_x(X)
for i in range(len(predictions)):
    print(f"Real: {y[i]} - Predicted: {predictions[i]}")