import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Imputer, LabelEncoder, OneHotEncoder, MinMaxScaler, StandardScaler
from matplotlib.colors import ListedColormap

df_wine = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data', header=None)
df_wine.columns = ['Class label', 'Alcohol', 'Malic Acid', 'Ash', 'Alcalinity of ash', 'Magnesium',
                   'Total phenols', 'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins',
                   'Color intensity', 'Hue', 'OD280/OD315 of diluted wines', 'Proline']

print('Class labels', np.unique(df_wine['Class label']))

print(df_wine.head())

X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values
"""The values attribute is used to assign the numpy array reprsentation of the feature columns and class 
labels to X and y
"""

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

"""Two common approaches to feature scaling are:
1. Normalization
2. Standardizatin

In normalization, the features are scaled so that they are in the range [0, 1], which is a special case of 
min-max scaling. This is done by:

xi = (xi - xmin)/(xmax-xmin)

where, 

xi= a particular sample
xmin = the smallest value in the feature column
xmax = the largest value in the feature column

Scikit-learn has an implementation of this procedure which can be used as follows:
"""

mms = MinMaxScaler()

X_train_norm = mms.fit_transform(X_train)
X_test_norm = mms.transform(X_test)

"""Although normalization via min-max scaling is useful  when we need values in a bounded interval, 
standard scaling can be more practical for many machine learning algorithms. Many linear models, such as 
logistic regression and SVM initialize weights to 0 or small random values close to 0. Using 
standardization, we center the feature columns at mean 0 with standard deviation 1 so that the feature 
columns take the form of a normal distribution which makes it easier to learn the weights. Standardization 
also maintains useful information about outliers and makes the algorithm less sensitive to them as opposed 
to min-max scaling which scales the data to a limited range of values. Standardization can be expressed by 
the following equation:

xi = (xi - xmean)/xstd

where,

xi = a particular sample
xmean = mean of a particular feature column
xstd = standard deviation of the feature column

Scikit-learn's implementation of standardization can be used as follows:
"""

sc = StandardScaler()

X_train_std = sc.fit_transform(X_train)
X_test_std = sc.transform(X_test)

"""For regularized models in scikit-learn that support L1 regularization, we can simply set the penalty 
parameter to 'l1' to yield the sparse solution.
"""

lr = LogisticRegression(penalty='l1', C=0.1)

lr.fit(X_train_std, y_train)
print('Training Accuracy: ', lr.score(X_train_std, y_train))
print('Test accuracy: ', lr.score(X_test_std, y_test)) #score method returns accuracy

print(lr.intercept_) #intercept_method returns intercepts between L1 penalty boundary and cost function ellipses

print(lr.coef_) #coef_ method returns weight vectors for each class

"""The weight vectors are sparse, i.e. they only have a few non-zero entries. As a result of L1 
regularization we've trained a model that is robust to the potentially irrelevant features in the dataset.
Now, we plot the regularization path, which is the weight coefficients of the different features for 
different regularization strengths.
"""

fig = plt.figure()
ax = plt.subplot(111)
colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black', 'pink', 'lightgreen', 'lightblue',
          'gray', 'indigo', 'orange']

weights, params = [], []

for c in np.arange(-4, 6):
    lr = LogisticRegression(penalty='l1', C=10**c, random_state=0)
    lr.fit(X_train_std, y_train)
    weights.append(lr.coef_[1])
    params.append(10**c)

weights = np.array(weights)

for column, color in zip(range(weights.shape[1]), colors):
    plt.plot(params, weights[:, column], label=df_wine.columns[column+1], color=color)

plt.axhline(0, color='black', linestyle='--', linewidth=3)
plt.xlim([10**(-5), 10**(5)])
plt.ylabel('weight coefficient')
plt.xlabel('C')
plt.xscale('log')
plt.legend(loc='upper left')
ax.legend(loc='upper center', bbox_to_anchor=(1.38, 1.03), ncol=1, fancybox=True)
plt.show()

"""As can be seen from the graph, all feature weights will be zero if we penalize the model with a strong 
regularization parameter (C<0.1); C is the inverse of regularization parameter lambda.
"""