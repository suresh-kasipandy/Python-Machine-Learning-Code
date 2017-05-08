import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

df = pd.DataFrame([['green', 'M', 10.1, 'class1'], ['red', 'L', 13.5, 'class2'], ['blue', 'XL', 15.3, 'class1']])

df.columns = ['Color', 'Size', 'Price', 'Class']

"""Here, color is a nominal feature as it has no implication of order while size which does is an 
ordinal feature. Price is a numerical feature. Previously practised learning algorithms for classification
do not use ordinal information in class labels.
"""

print("Data Frame")
print(df)

"""To make sure that the learning algorithm implements the ordinal features correctly, we need to replace 
the categorical string values with integers. However, there is no convenient function that automatically 
derives the correct order of the labels of the 'Size' feature. Thus, this needs to be mapped manually.
"""

size_mapping = {'M' : 1, 'L' : 2, 'XL' : 3}

df['Size'] = df['Size'].map(size_mapping)

print('\nData Frame with size mapping')
print(df)


"""To transform the integer values back to the original string values we can define a 
reverse-mapping dictionary and use the pandas' 'map' method on the transformed feature column.
"""
"""
inv_size_mapping = {v : k for k, v in size_mapping.items()}

df['Size'] = df['Size'].map(inv_size_mapping)

print('\nData Frame with size mapping reversed')
print(df)
"""
"""While many scikit-learn estimators convert class labels to integers internally, it is good practice to
provide class labels as integer arrays to avoid technical glitches. As class labels are not ordinal and 
hence it doesn't matter what value is assigned to a particular string-label.
"""

class_mapping = {cl : idx for idx, cl in enumerate(np.unique(df['Class']))}

df['Class'] = df['Class'].map(class_mapping)

print('\nData Frame with class mapping')
print(df)

"""The key-value pairs in the mapping dictionary can be reversed to switch the connverted class labels 
back to their original string representation.
"""
"""
inv_class_mapping = { v : k for k, v in  class_mapping.items()}

df['Class'] = df['Class'].map(inv_class_mapping)

print('\nData Frame with class mapping reversed')
print(df)
"""
"""Alternatively, the LabelEncoder class directly implemented in scikit achieves the same.
"""

class_le = LabelEncoder()
y = class_le.fit_transform(df['Class'].values)

print("\n", y)


"""The fit_transform method is a shortcut for calling fit and transform separately. The inverse_transform 
method can be used to switch the integer class labels back to their original string representation"""

#print("\n", class_le.inverse_transform(y))

"""However we can't use this approach for nominal features like 'Color' in this case, as it would make 
the learning algorithm assume that one color is greater than another which is an incorrect assumption. 
While the algorithm could still produce useful results, these results would not be optimal. In cases with 
nominal features we use one-hot encoding instead."""

color_le = LabelEncoder()

ohe = OneHotEncoder(categorical_features=[0])

X = df[['Color', 'Size', 'Price']].values
X[:, 0] = color_le.fit_transform(X[:, 0])

print(ohe.fit_transform(X).toarray())

"""When we initialized the OneHotEncoder, we defined the column position of the  of the variable that we 
want to transform via the categorical_features parameter. By default, the OneHotEncoder returns a sparse 
matrix when we use the transform/fit_transform method which we converted to a regular (dense) NumPy 
array via the toarray method. Sparse matrices are simply a more efficient way of storing large datasets, 
and one that is supported by many scikit-leanr functions, which is especially useful if  it contains a lot 
of zeros. To omit the toarray() step we could initialize the encoder as 'OneHotEncoder(..., sparse=False)' 
to return a regular NumPy array.
"""

"""The method get_dummies implemented in pandas provides an even more convenient way to create those dummy 
features using one-hot encoding. Applied on a DataFrame, the get_dummies method will only convert string 
columns and leave all other columns unchanged.
"""

print(pd.get_dummies(df[['Price', 'Color', 'Size']]))
