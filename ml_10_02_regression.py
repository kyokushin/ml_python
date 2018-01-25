import numpy as np
import utils
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

df = utils.read_csv('./housing.data', url='https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data', sep='\s+')

df.columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX',
              'RM', 'AGE', 'DIS', 'RAD', 'TAX',
              'PTRATIO', 'B', 'LSTAT', 'MEDV']
cols = ['LSTAT', 'INDUS', 'NOX', 'RM', 'MEDV']
sns.set(style='whitegrid', context='notebook')

cm = np.corrcoef(df[cols].values.T)

X = df[['RM']].values
y = df[['MEDV']].values

sc_x = StandardScaler()
sc_y = StandardScaler()

X_std = sc_x.fit_transform(X)
y_std = sc_y.fit_transform(y)

lr = utils.LinearRegressionGD()

lr.fit(X_std, y_std)

plt.plot(range(1, lr.n_iter+1), lr.cost_)
plt.ylabel('SSE')
plt.xlabel('Epoch')
plt.show()

utils.lin_regplot(X_std, y_std, lr)
plt.xlabel('Average number of rooms [RM] (standardized)')
plt.ylabel('Price in $1000\'s [MEDV] (standardized)')
plt.show()
