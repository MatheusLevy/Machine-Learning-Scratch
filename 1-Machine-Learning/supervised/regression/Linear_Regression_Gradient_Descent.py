import numpy as np

# Gradient Descent 
# Let's say we have a hypoteses function h(x) that depends on a set of parameters theta and a input x
# So h(x) need to predict a value closest to the real value y
# So to find the best parameters theta we need to minimize the cost function J(theta)
# J(theta) is described as the mean squared error between the predicted value and the real value
# J(theta) = 1/2m * sum(h(x) - y)^2
# The 1/2 is used to simplify the derivative of the cost function
# So let's minimize J(theta) using the gradient descent algorithm
# The fastest form is using the normal equation of the derivative of J(theta)
# Or we can use the gradient descent algorithm that is a iterative algorithm that update the parameters theta
# theta_j := theta_j - alpha * dJ(theta)/dtheta
# Ou seja theta_j = theta_j - alpha * 1/m * sum(h(x) - y) * x_j
# To a better understaning of how to derivate this equations see:
# https://www.youtube.com/watch?v=4b4MUYve_U8

class Linear_Regression():
    def __init__(self, X, y, alpha, num_iter):
        self.theta = np.zeros((X.shape[1] + 1, 1)) # theta is a vector of parameters for each feature plus one for the bias
        self.alpha = alpha
        self.num_iter = num_iter
        self.X = X
        self.y = y

    def h_x(self, X):
        ones = np.ones((X.shape[0], 1))
        concat = np.concatenate([ones, X], axis=1)
        return np.dot(concat, self.theta)

    def loss (self, X, y):
        m = X.shape[0] # num of Samples
        h = self.h_x(X) # prediction for each sample as a vector
        return 1/(2*m) * np.sum(np.square(h - y)) # J(theta) = 1/2m * sum(h(x) - y)^2

    def gradient_descent(self):
        m = self.X.shape[0]
        for _ in range(self.num_iter):
            error = self.h_x(self.X) - self.y
            print
            # Att weights 
            self.theta[0] = self.theta[0] - self.alpha * 1/m * np.sum(error)
            self.theta[1:] = self.theta[1:] - self.alpha * 1/m * np.dot(self.X.T, error)

            print(self.loss(self.X, self.y))

        
X = np.array([[1,4], [2,8], [3,12], [4,16]])
y=  np.array([[2], [4], [6], [8]])

lr = Linear_Regression(X, y, 0.01, 1000)
lr.gradient_descent()
print(lr.h_x(np.array([[5, 20]])))