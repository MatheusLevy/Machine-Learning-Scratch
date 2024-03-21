import numpy as np

#  Naive Bayes
#   Event A: The event we want to predict
#   Event B: The event that was observed
#   p(A|B) = The probability of event A happen if event B has happened
#   p(B|A) = The probability of event B happen if event A has happened
#   p(A) = The probability of event A happen
#   p(B) = The probability of event B happen
#   
#   Bayes Theorem:
#   P(A|B) = P(B|A) * P(A) / P(B)
#
#   p(B) = p(B|A) * p(A) + p(B|A') * p(A') where A' when A has not happened
#   Bayes Theorem:
#   P(A|B) = P(B|A) * P(A) / (P(B|A) * P(A) + P(B|A') * P(A'))
#

class Naive_Bayes:
    def __init__(self, X, y):
        self.num_exemples, self.num_features = X.shape
        self.num_classes = len(np.unique(y))
        self.eps = 1e-6

    def fit(self, X,y):
        self.classes_mean = {}
        self.classes_variance = {}
        self.classes_prior = {}
        for c in range(self.num_classes):
            X_c = X[y==c]
            self.classes_mean[str(c)] = np.mean(X_c, axis=0)
            self.classes_variance[str(c)] = np.var(X_c, axis=0)
            self.classes_prior[str(c)] = X_c.shape[0] / X.shape[0]


    def predict(self, X):
        probs = np.zeros((self.num_exemples, self.num_classes))
        for c in range(self.num_classes):
            prior = self.classes_prior[str(c)]
            probs_c = self.density(X, self.classes_mean[str(c)], self.classes_variance[str(c)])
            probs[:, c] = probs_c + np.log(prior)
        return np.argmax(probs, 1)

    def density(self, x, mean, sigma):
        # (2pi)^(k/2)det(sum)^(-1/2)exp(-1/2(x-mu)^Tsum^(-1)(x-mu))
        # Calculate probability from Gaussian density function
        const = -self.num_features / 2 * np.log(2 * np.pi) - 0.5 * np.sum(
            np.log(sigma + self.eps)
        )
        probs = 0.5 * np.sum(np.power(x - mean, 2) / (sigma + self.eps), 1)
        return const - probs
    
# X = np.loadtxt(r'data\data.txt', delimiter=',')
# y = np.loadtxt(r'data\targets.txt')-1
X = np.load('data/X.npy')
y = np.load('data/y.npy')
print(X.shape, y.shape)
nb = Naive_Bayes(X, y)
nb.fit(X, y)
print(nb.predict(X))
print(sum(nb.predict(X) == y)/y.shape[0])
print(sum(y)/y.shape[0])