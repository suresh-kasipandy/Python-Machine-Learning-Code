import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from matplotlib.colors import ListedColormap
iris = datasets.load_iris()


class Perceptron(object):
    """
    Perceptron classifier
    
    Parameters
    ------------
    eta : float
        Learning rate (between 0.0 and 1.0)
    n_iter : int
        Epochs or number of passes over training set
    
    Attributes
    ------------
    w_ : 1-d array
        Weights after fitting
    errors_ : list
        Number of misclassifications after each epoch
    """

    def __init__(self, eta, n_iter):
        self.eta=eta
        self.n_iter=n_iter

    def fit(self, X, y):
        """
         Fits training data
         
         Parameters
         ------------
         X : {array-like}, shape = [n_samples, n_features]
            Training vectors where n_samples is the number of samples and n_features is the number of features
         y : array-like, shape=[n_samples]
            Target values
        
        Returns
        ------------
        self : object
        """
        self.w_=np.zeros(X.shape[1]+1)
        self.errors_=[]
        for _ in range(self.n_iter):
            errors=0
            for xi, target in zip(X, y):
                update=self.eta*(target-self.predict(xi))
                self.w_[1:]+=update*xi
                self.w_[0]+=update
                errors+=int(update!=0.0)
            self.errors_.append(errors)

    def net_input(self, X):
        """Returns net input"""
        return(np.dot(self.w_[1:], X)+self.w_[0])
    def predict(self, X):
        """Returns predicted class label"""
        return(np.where(self.net_input(X)>=0, 1, -1))

"""Load dataset using sklearn"""
X=iris.data[:100, [0,2]]
y=iris.target
print("sklearn")
print(X)
print(y)

"""Load dataset using pandas"""
df=pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data", header=None)
X=df.iloc[:100, [0,2]].values
y=df.iloc[:100, 4].values
y=np.where(y=='Iris-setosa', -1, 1)
print("pandas")
print(X)
print(y)

plt.scatter(X[:50, 0], X[:50, 1], color="red", label ="setosa", marker = "o")
plt.scatter(X[50:100, 0], X[50:100, 1], color="blue", label = "versicolor", marker = "^")
plt.xlabel("sepal length")
plt.ylabel("petal length")
plt.legend(loc="upper left")
plt.show()

ppn=Perceptron(eta = 0.01, n_iter=50)
ppn.fit(X, y)
print("Weights: ", ppn.w_)
print("Errors: ", ppn.errors_)
plt.xlabel("Epochs")
plt.ylabel("Number of misclassifications")
plt.plot(range(1, len(ppn.errors_)+1), ppn.errors_, marker='o')
plt.show()

def plot_decision_regions(X, y, classifier, resolution = 0.02):

    #setup marker generator and color map
    colors = ('black', 'purple')
    markers = ('^', 'v')

    cmap = ListedColormap(colors[:len(np.unique(y))])

    x1_min, x1_max = X[:, 0].min()-1, X[:, 0].max()+1
    x2_min, x2_max = X[:, 1].min()-1, X[:, 1].max()+1

    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution))

    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]))
    Z = Z.reshape(xx1.shape)

    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    #plot class samples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y==cl, 0], y=X[y==cl, 1], c=cmap(idx), marker=markers[idx], label=cl)

plot_decision_regions(X, y, ppn)

plt.xlabel("Sepal Length (cm.)")
plt.ylabel("Petal Length (cm.)")
plt.legend(loc="upper left")

plt.show()
