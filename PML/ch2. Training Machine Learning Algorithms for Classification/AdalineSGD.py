import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import datasets
from numpy.random import seed

class AdalineSGD(object):
    """ADAptive LInear NEuron classifier with stochastic gradient descent
    
    Parameters
    ------------
    eta : float
        Learning rate (between 0.0 and 1.0)
        
    n_iter : int
        Epochs or number of passes over training data
    
    Attributes
    ------------
    w_ : 1-d array
        Weights after fitting
    errors_ : list
        Number of errors in each misclassification
    shuffle : bool (default : True)
        Shuffles training data every epoch if true to prevent cycles
    random_state : int (default : None)
        Set random state for shuffling and initializing weights
    """

    def __init__(self, eta=0.01, n_iter=10, shuffle=True, random_state=None):
        self.eta = eta
        self.n_iter = n_iter
        self.shuffle = shuffle
        self.w_initialized = False

        if(random_state):
            seed(random_state)

    def fit(self, X, y):
        """Fit training data
        
        Parameters
        ------------
        X : {array-like}, shape = [n_samples, n_features]
            Training vectors where n_samples is number of samples and n_features is number of features
        y : array-like
            Target values
        
        Returns
        ------------
        self : object
        """

        self._initialize_weights(X.shape[1])
        self.cost_ = []
        for i in range(self.n_iter):
            if self.shuffle:
                X, y = self._shuffle(X, y)
            cost = []
            for xi, target in zip(X, y):
                cost.append(self._update_weights(xi, target))
            avg_cost = sum(cost)/len(y)
            self.cost_.append(avg_cost)

        return self

    def partial_fit(self, X, y):
        """Fit training data without reinitializing the weights"""

        if not self.w_initialized:
            self._initialize_weights(X.shape[1])
        if y.ravel().shape[0] > 1:
            for xi, target in zip(X, y):
                self._update_weights(xi, target)
        else:
            self._update_weights(X, y)
        return self

    def _shuffle(self, X, y):
        """Shuffle training data"""

        r = np.random.permutation(len(y))

        return X[r], y[r]

    def _initialize_weights(self, m):
        """Initialize weights to zeros"""

        self.w_ = np.zeros(m+1)
        self.w_initialized = True

    def _update_weights(self, xi, target):
        """Apply ADALINE learning rule to update the weights"""

        output = self.net_input(xi)
        error = target - output
        self.w_[1:] += self.eta*xi.dot(error)
        self.w_[0] += self.eta*error
        cost = 0.5*error**2
        return cost

    def net_input(self, X):
        """Returns net input"""
        return np.dot(X, self.w_[1:])+self.w_[0]

    def activation(self, X):
        """Returns linear activation"""
        return self.net_input(X)

    def predict(self, X):
        """Return class label after unit step"""
        return np.where(self.activation(X)>=0, 1, -1)

def plot_decision_regions(X, y, classifier, resolution = 0.02):

    #setup marker generator and color map
    markers = ('^', 'v', 'o', '*', 'x')
    colors = ('red', 'blue', 'green', 'yellow', 'cyan')

    cmap = ListedColormap(colors[:len(np.unique(y))])

    #plot decision surfaces
    x1_min, x1_max = X[:, 0].min()-1, X[:, 0].max()+1
    x2_min, x2_max = X[:, 1].min()-1, X[:, 1].max()+1

    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution))

    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)

    plt.contourf(xx1, xx2, Z, alpha = 0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    #plot class samples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(X[y==cl, 0], X[y==cl, 1], alpha=0.8, c=cmap(idx), marker=markers[idx], label=cl)

ada = AdalineSGD(n_iter=15, eta=0.01, random_state=1)
iris = datasets.load_iris()
X = iris.data[:100, [0,2]]
y = iris.target[:100]
y = np.where(y==0, 1, -1)

X_std = np.copy(X)
X_std[:, 0] = (X[:, 0] - X[:, 0].mean())/X[:, 0].std()
X_std[:, 1] = (X[:, 1] - X[:, 1].mean())/X[:, 1].std()


ada.fit(X_std, y)
plot_decision_regions(X_std, y, classifier=ada)
plt.title('ADALINE with Stochastic Gradient Descent')
plt.xlabel('Sepal Length [standardized]')
plt.ylabel('Petal Length [standardized]')
plt.legend(loc='upper left')
plt.show()
plt.plot(range(1, len(ada.cost_)+1), ada.cost_, marker ='o')
plt.xlabel('Epochs')
plt.ylabel('Average cost')
plt.show()