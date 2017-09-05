import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.base import clone
from itertools import combinations
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
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

feat_labels = df_wine.columns[1:]
forest = RandomForestClassifier(n_estimators=10000, random_state=0, n_jobs=-1)
forest.fit(X_train, y_train)

importances = forest.feature_importances_

indices = np.argsort(importances)[::-1] # ソート結果のインデックス

for f in range(X_train.shape[1]):
	print('%2d %-*s %s' % (f + 1, 30, feat_labels[indices[f]],
	                       importances[indices[f]]))

plt.title('Feature Importances')
plt.bar(range(X_train.shape[1]), importances[indices],
        color='lightblue', align='center')
plt.xticks(range(X_train.shape[1]), feat_labels[indices], rotation=90)
plt.xlim([-1, X_train.shape[1]])
plt.tight_layout()
plt.show()

X_selected = X_train[:, forest.feature_importances_ > 0.15]
print(X_selected.shape)
