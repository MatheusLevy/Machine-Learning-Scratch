import numpy as np


# X : (Training_exemples, Features)
# y : (Training_exemples, 1)
# output = (Features, 1)



def linear_regression_normal_equation(X,y):
    ones = np.ones((X.shape[0], 1))
    X = np.append(ones, X, axis=1)
    W = np.dot(np.linalg.pinv(np.dot(X.T,X)), np.dot(X.T,y))
    return W

X = np.array([[2.28835, 3.4148, 2]]).T
y = np.array([-2.54857, -2.80671, -1])
print(X.shape, y.shape)
print(linear_regression_normal_equation(X, y))
