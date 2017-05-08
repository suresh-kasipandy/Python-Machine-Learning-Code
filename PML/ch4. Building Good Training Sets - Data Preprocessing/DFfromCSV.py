import pandas as pd
from io import StringIO
csv_data = '''A,B,C,D
1.0,2.0,3.0,4.0
5.0,6.0,,8.0
10.0,11.0,12.0,'''

"""If you are using Python 2.7 you need to convert the string to unicode:

csv_data = unicode(csv_data)"""

#StringIO allows us to read the string csv_data as if it were regular local csv file
df = pd.read_csv(StringIO(csv_data))

print(df)

#The sum function can be used in the following way to count the number of missing values per column
print(df.isnull().sum())

#The underlying numpy array of the DataFrame can always be accessed via the values attribute
print(df.values)

#The dropna() function can be used to drop rows with missing values
print(df.dropna())

#Similarly we can drop columns with at least one NaN by setting the axis argument to 1
print(df.dropna(axis=1))

"""The dropna() method supports several additional parameters that can come in handy"""

#only drops rows where all columns are NaN
print(df.dropna(how='all'))

#drops rows that don't have at least 4 non-Nan values
print(df.dropna(thresh=4))

#only drops rows where NaN appears in specific columns
df.dropna(subset=['C'])
