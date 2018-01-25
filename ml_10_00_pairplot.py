import utils
import matplotlib.pyplot as plt
import seaborn as sns

df = utils.read_csv('./housing.data', url='https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data', sep='\s+')
df.columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX',
              'RM', 'AGE', 'DIS', 'RAD', 'TAX',
              'PTRATIO', 'B', 'LSTAT', 'MEDV']

sns.set(style='whitegrid', context='notebook')
cols = ['LSTAT', 'INDUS', 'NOX', 'RM', 'MEDV']

sns.pairplot(df[cols], size=2.5)
plt.show()
