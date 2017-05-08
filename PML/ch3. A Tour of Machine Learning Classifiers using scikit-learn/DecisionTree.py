from sklearn.tree import DecisionTreeClassifier
import pandas as pd
from numpy.random import seed
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn import datasets
from sklearn.tree import export_graphviz

tree = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=0)
iris = datasets.load_iris()

X = iris.data[:, [2,3]]
y = iris.target

sc = StandardScaler()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

#sc.fit(X_train)

#X_train_std = sc.transform(X_train)
#X_test_std = sc.transform(X_test)

#While feature scaling might be desired for visualization purposed it is not reuired for decision trees

X_combined_std = np.vstack((X_train, X_test))
y_combined = np.hstack((y_train, y_test))

def plot_decision_regions(X, y, classifier, resolution=0.02, test_idx=None):

    #setup marker generator and color map
    markers = ('s', 'x', '^', 'v', '*')
    colors = ('blue', 'red', 'green', 'yellow', 'black')

    cmap = ListedColormap(colors[:len(np.unique(y))])

    #plot decision surface
    x1_min, x1_max = X[:, 0].min()-1, X[:, 0].max()+1
    x2_min, x2_max = X[:, 1].min()-1, X[:, 1].max()+1

    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution))

    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)

    Z = Z.reshape(xx1.shape)

    plt.contourf(xx1, xx2, Z, cmap=cmap, alpha=0.4)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    #plot all samples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(X[y==cl, 0], X[y==cl, 1], c=cmap(idx), marker=markers[idx], label=cl, alpha=0.8)

    #highlight test samples
    if test_idx:
        X_test, y_test = X[test_idx, :], y[test_idx]
        plt.scatter(X_test[:, 0], X_test[:, 1], marker='o', s=55, linewidths=1.0, c='', label='Test samples')

tree.fit(X_train, y_train)

plot_decision_regions(X_combined_std, y_combined, classifier=tree, test_idx=range(105, 150))
plt.xlabel('Petal Length [cm.]')
plt.ylabel('Petal Width [cm.]')
plt.legend(loc='upper left')
plt.show()

export_graphviz(tree, out_file='tree.dot', feature_names=['petal length', 'petal width'])