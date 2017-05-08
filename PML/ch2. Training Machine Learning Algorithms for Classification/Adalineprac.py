import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import datasets

class ADALINE(object):
    """
    ADAptive LInear NEuron classfifier
    
    Parameters
    ------------
    eta : float
        Learning rate (between 0.0 and 1.0)
    n_iter : int
        Epochs or number of iterations
    
    Attribues
    ------------
    w_ : 1-d array
        Weights after fitting
    cost_ : list
        Value of cost function after each epoch
    """

    def __init__(self, eta = 0.01, n_iter = 100):
        #initializes parameters
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X, y):
        #Fits the training data
        self.w_= np.zeros(1 + X.shape[1])
        self.cost_ = []
        for _ in range(self.n_iter):
            errors = y - self.net_input(X)
            self.w_[1:] += self.eta*np.dot(errors, X)
            self.w_[0] += self.eta*errors.sum()
            cost = (errors**2).sum()/2.0
            self.cost_.append(cost)
        return self

    def net_input(self, X):
        #Returns the net input
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def activation(self, X):
        return self.net_input(X)

    def predict(self, X):
        return np.where(self.activation(X)>=0.0, 1, -1)

iris = datasets.load_iris()

X = iris.data[:100, [0, 2]]
y = iris.target[:100]
y=np.where(y==0, 1, -1)
plt.scatter(X[:50, 0], X[:50, 1], marker = '^', color = 'red', label = 'Iris-Setosa')
plt.scatter(X[50:100, 0], X[50:100, 1], marker ='v', color ='blue', label = 'Iris-Versicolor')
plt.xlabel('Sepal Length')
plt.ylabel('Petal Length')
plt.legend(loc = 'upper left')
plt.show()

ADA = ADALINE(eta=0.0001, n_iter=400)

ADA.fit(X, y)
print(min(ADA.cost_))
plt.plot(range(1, len(ADA.cost_)+1), ADA.cost_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Cost function')
plt.show()

fig, ax = plt.subplots(nrows=1, ncols=2, figsize =(8, 4))

ada1=ADALINE(eta=0.01, n_iter= 10).fit(X, y)
ax[0].plot(range(1, len(ada1.cost_)+1), np.log10(ada1.cost_), marker ='o')
ax[0].set_xlabel('Epochs')
ax[0].set_ylabel('Cost function')
ax[0].set_title('ADALINE Learning rate : 0.01')

ada2=ADALINE(eta=0.0001, n_iter= 10).fit(X, y)
ax[1].plot(range(1, len(ada2.cost_)+1), ada2.cost_, marker ='o')
ax[1].set_xlabel('Epochs')
ax[1].set_ylabel('Cost function')
ax[1].set_title('ADALINE Learning rate : 0.0001')

plt.show()

X_std = X.copy()

X_std[:, 0] = (X[:, 0] - X[:, 0].mean())/X[:, 0].std()
X_std[:, 1] = (X[:, 1] - X[:, 1].mean())/X[:, 1].std()

ada3 = ADALINE(n_iter = 10, eta = 0.01).fit(X_std, y)

def plot_decision_regions(X, y, classifier, resolution = 0.02):
    #Create color map and marker generator
    colors = ('red', 'blue', 'green', 'black', 'yellow')
    markers = ('^', 'v', 'o', '*', '-')

    cmap = ListedColormap(colors[:len(np.unique(y))])

    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution))

    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)

    Z = Z.reshape(xx1.shape)

    plt.contourf(xx1 ,xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y==cl, 0], y=X[y==cl,1],alpha=0.8, marker=markers[idx], c=cmap(idx), label=cl)


plot_decision_regions(X_std, y, ada3)

plt.title('ADALINE Gradient Descent')
plt.xlabel('Sepal Length')
plt.ylabel('Petal Length')
plt.legend(loc='upper left')
plt.show()