import pandas as pd
from io import StringIO
from sklearn.preprocessing import Imputer

csv_data='''A,B,C,D
1.0,2.0,3.0,4.0
5.0,6.0,,8.0
10.0,11.0,12.0,'''

df = pd.read_csv(StringIO(csv_data))

imr = Imputer(missing_values='NaN', strategy='mean', axis=0)

"""Here we replaced each NaN value by the corresponding mean which is separately calculated for each 
feature column. If we switched axis=0 to axis=1 we'd calculate row means instead. The other options for 
the strategy argument are 'median' and 'most_frequent' with the latter replacing the missing values with 
the most frequent values which is useful for imputting values in the case of categorical features."""

imr.fit(df)
imputed_values = imr.transform(df.values)
print(imputed_values)