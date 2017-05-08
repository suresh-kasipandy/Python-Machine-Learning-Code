import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class AdalineGD(object):
    """ADAptiver LInear NEuron classifier

    Parameters
    ------------
    eta = float
        Learning rate (between 0.0 and 1.0)
    n_iter = int
        Number of passes over training dataset

    Attributes
    ------------
    w_ : 1-d array
        Weights after fitting
    cost_ : list
        Value of cost function after each epoch

    """
    def __init__(self, eta=0.01, n_iter=50):
        self.eta=eta
        self.n_iter=n_iter

    def fit(self, X, y):
        """Fits training data

        Parameters
        ------------
        X : {array-like}, shape=[n_samples, n_features]
            Training vectors where n_samples is the number of samples and n_features is the number of features.

        y : array-like, shape=[n_samples]
            Target values.

        Returns
        ------------
        self : object

        """
        self.w_=np.zeros(1+X.shape[1])
        self.cost_=[]

        for i in range(self.n_iter):
            errors=y-self.net_input(X)
            print("X:", X, X.T, "errors:", errors, X.T.dot(errors))
            self.w_[1:]+=self.eta*np.dot(X.T, errors)
            self.w_[0]+=self.eta*errors.sum()
            cost=(errors**2).sum()/2.0
            self.cost_.append(cost)
        return self

    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def activation(self, X):
        return self.net_input(X)
    def predict(self, X):
        return np.where(self.activation(X)>=0, 1, -1)
df=pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data", header=None)
print(df.tail())
X=df.iloc[:100, [0, 2]].values
y=df.iloc[:100, 4].values
y=np.where(y=='Iris-setosa', -1, 1)
print(X)
print(y)
ADA=AdalineGD(eta=0.1, n_iter=10)
ADA.fit(X, y)
print(ADA.w_)
print(ADA.cost_)
plt.scatter(X[:50, 0], X[:50, 1], color='red', marker='x', label='setosa')
plt.scatter(X[50:100,0], X[50:100, 1], color='blue', marker='o', label='versicolor')
plt.xlabel('sepal length')
plt.ylabel('petal length')
plt.legend(loc='upper left')
plt.show()
fig, ax=plt.subplots(nrows=1, ncols=2, figsize=(8,4))
ada1=AdalineGD(eta=0.01, n_iter=10).fit(X, y)
ax[0].plot(range(1, len(ada1.cost_)+1), np.log10(ada1.cost_), marker='o')
ax[0].set_xlabel('Epochs')
ax[0].set_ylabel('Log Sum Squared Error')
ax[0].set_title('ADALINE learning rate 0.01')
ada2=AdalineGD(eta=0.0001, n_iter=10).fit(X, y)
ax[1].plot(range(1, len(ada2.cost_)+1), np.log10(ada2.cost_), marker='x')
ax[1].set_xlabel('Epochs')
ax[1].set_ylabel('log sum squared error')
ax[1].set_title('ADALINE learning rate 0.0001')
plt.show()

X_std=np.copy(X)
X_std[:,0]=(X[:,0]-X[:,0].mean())/X[:,0].std()
X_std[:,1]=(X[:,1]-X[:,1].mean())/X[:,1].std()
ada=AdalineGD(eta=0.01, n_iter=15)
ada.fit(X_std, y)
plt.plot(range(1, len(ada.cost_)+1), ada.cost_, marker='o', color='blue')
plt.xlabel('Epochs')
plt.ylabel('ADA cost')
plt.title('Adaline - gradient descent (standardized) ')
plt.show()