import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
import utils

df_wine = utils.read_csv('wine.data',
                         'https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data')
df_wine.columns = ['Class label', 'Alchol', 'Malic acid', 'Ash',
                   'Alcalinity of Ash', 'Magnesium', 'Total phenols',
                   'Flavonoids', 'Nonflavanoid phenols', 'Proanthocyanins',
                   'Color intensity', 'Hue', '0D280/0D315 of diluted wines',
                   'Proline']

print('Class labels', np.unique(df_wine['Class label']))
df_wine.head()

X = df_wine.iloc[:, 1:].values
y = df_wine.iloc[:, 0].values

print("Class labels:", np.unique(y))

X_train, X_test, y_train, y_test = train_test_split(
	X, y, test_size=0.3, random_state=0)

sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))

pca = PCA(n_components=2)
lr = LogisticRegression()

X_combined_pca = pca.fit_transform(X_combined_std)
X_train_pca = pca.transform(X_train_std)

lr.fit(X_train_pca, y_train)

utils.plot_decision_regions(X_combined_pca, y_combined, classifier=lr,
                            test_idx=range(X_train.shape[0],
                                           X_train.shape[0] + X_test.shape[0]))
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend(loc='lower left')
plt.show()
