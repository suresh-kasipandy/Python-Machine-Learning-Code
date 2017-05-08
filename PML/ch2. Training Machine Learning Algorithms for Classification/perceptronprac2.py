import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import datasets

class Perceptron(object):
    """Perceptron classifier
    
    Parameters
    ------------
    eta : float
        Learning rate
    n_iter : int
        Number of iterations or Epochs
    
    Attributes
    ------------
    w_ : 1-d array
        Weights after fitting
    errors_ : list
        Number of misclassifications after each epoch
    """
    def __init__(self, eta=0.01, n_iter=10):
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X, y):
        """Fits the training data
        
        Parameters
        ------------
        X : {array-like}, shape = [n_samples, n_features]
            Training vectors where n_samples is number of samples and n_features is number of features
        y : array-like, shape = [n_samples]
            Target values
        
        Returns
        ------------
        self : object
        """
        self.w_ = np.zeros(X.shape[1]+1)
        self.errors_ = []

        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                update = self.eta*(target - self.predict(xi))
                self.w_[1:] += update*xi
                self.w_[0] += update
                errors += int(update!=0)
            self.errors_.append(errors)

    def net_input(self, X):
        #Returns net input
        return(np.dot(self.w_[1:], X) + self.w_[0])

    def predict(self, X):
        #Predicts class label
        return(np.where(self.net_input(X)>=0, 1, -1))

iris = datasets.load_iris()

X = iris.data[:100, [0,2]]
y = iris.target[:100]
y = np.where(y==0, 1, -1)
print(y)
plt.scatter(X[:50, 0], X[:50, 1], marker = '^', color = 'purple', label = 'iris-setosa')
plt.scatter(X[50:100, 0], X[50:100, 1], marker = 'v', color = 'violet', label = 'iris-versicolor')
plt.xlabel('sepal length (cm.)')
plt.ylabel('petal length (cm.)')
plt.legend(loc='upper left')
plt.show()

ppn = Perceptron()

ppn.fit(X, y)

plt.plot(range(1, len(ppn.errors_)+1), ppn.errors_, marker = 'o')
plt.xlabel('Epochs')
plt.ylabel('Misclassifications')
plt.show()

def plot_decision_regions(X, y, classifier, resolution = 0.02):

    #setup color map and marker generator
    colors = ('red', 'blue', 'green', 'yellow', 'violet')
    markers = ('^', 'v', '*', 'o', '-')

    cmap = ListedColormap(colors[:len(np.unique(y))])

    att1_min, att1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    att2_min, att2_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    att1s, att2s = np.meshgrid(np.arange(att1_min, att1_max, resolution), np.arange(att2_min, att2_max, resolution))

    Z = classifier.predict(np.array([att1s.ravel(), att2s.ravel()]))
    Z = Z.reshape(att1s.shape)
    plt.contourf(att1s, att2s, Z, alpha = 0.4, cmap=cmap)
    plt.xlim(att1s.min(), att1s.max())
    plt.ylim(att2s.min(), att2s.max())

    #plot samples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y==cl, 0], y=X[y==cl, 1], alpha=0.8, c=cmap(idx), marker=markers[idx], label=np.where(cl==1, 'Iris-Setosa', 'Iris-Versicolor'))

plot_decision_regions(X, y, ppn)
plt.xlabel('Sepal length (cm.)')
plt.ylabel('Petal length (cm)')
plt.legend(loc='upper left')
plt.show()

