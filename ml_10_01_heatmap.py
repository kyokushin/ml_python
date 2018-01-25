import numpy as np
import utils
import seaborn as sns
import matplotlib.pyplot as plt

df = utils.read_csv('./housing.data', url='https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data', sep='\s+')

df.columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX',
              'RM', 'AGE', 'DIS', 'RAD', 'TAX',
              'PTRATIO', 'B', 'LSTAT', 'MEDV']
cols = ['LSTAT', 'INDUS', 'NOX', 'RM', 'MEDV']
sns.set(style='whitegrid', context='notebook')

cm = np.corrcoef(df[cols].values.T)
sns.set(font_scale=1.5)

hm = sns.heatmap(cm, cbar=True, annot=True,
                 square=True, fmt='.2f',
                 annot_kws={'size': 15},
                 yticklabels=cols, xticklabels=cols)

plt.show()
