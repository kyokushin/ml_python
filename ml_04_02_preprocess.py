import pandas as pd
import numpy as np
from io import StringIO
from sklearn.preprocessing import Imputer

df = pd.DataFrame([
    ['green', 'M', 10.1, 'class1'],
    ['red', 'L', 13.5, 'class2'],
    ['blue', 'XL', 15.3, 'class1']
])
df.columns = ['color', 'size', 'price', 'classlabel']
print(df)

size_mapping = {'XL': 3, 'L': 2, 'M': 1}
df['size'] = df['size'].map(size_mapping)
print(df)

class_mapping = {label: idx for idx, label in enumerate(np.unique(df['classlabel']))}
print('class mapping:', class_mapping)

df['classlabel'] = df['classlabel'].map(class_mapping)
print(df)
