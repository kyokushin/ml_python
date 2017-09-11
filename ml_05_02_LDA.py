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

np.set_printoptions(precision=4)

mean_vecs = []
for label in range(1, 4):
	mean_vecs.append(np.mean(X_train_std[y_train == label], axis=0))
	print('MV %s: %s\n' % (label, mean_vecs[label - 1]))

d = X.shape[1]

S_W = np.zeros((d, d))
for label, mv in zip(range(1, 4), mean_vecs):
	class_scatter = np.cov(X_train_std[y_train == label].T)
	S_W += class_scatter

print('Scaled within-class scatter matrix:%sx%s'
      % (S_W.shape[0], S_W.shape[1]))

mean_overall = np.mean(X_train_std, axis=0)
S_B = np.zeros((d, d))
for i, mean_vec in enumerate(mean_vecs):
	n = X_train[y_train == i + 1, :].shape[0]
	mean_vec = mean_vec.reshape(d, 1)
	mean_overall = mean_overall.reshape(d, 1)
	S_B += n * (mean_vec - mean_overall).dot((mean_vec - mean_overall).T)

print('Between-class scatter matrix: %sx%s'
      % (S_B.shape[0], S_B.shape[1]))
print(S_B)

eigen_vals, eigen_vecs = np.linalg.eig(np.linalg.inv(S_W).dot(S_B))
eigen_pairs = [(np.abs(eigen_vals[i]), eigen_vecs[:, i]) for i in range(len(eigen_vals))]
eigen_pairs = sorted(eigen_pairs, key=lambda k: k[0], reverse=True)
print('Eigenvalues in decreasing order:\n')
for eigen_pair in eigen_pairs:
	print(eigen_pair[0])

tot = sum(eigen_vals.real)
discr = [(i / tot) for i in sorted(eigen_vals.real, reverse=True)]
cum_discr = np.cumsum(discr)
plt.bar(range(1, 14), discr, alpha=0.5, align='center',
        label='indivisual "discriminability"')
plt.step(range(1, 14), cum_discr, where='mid',
         label='cumulative "discriminability"')
plt.ylabel('"discriminability" ratio')
plt.xlabel('Linear Dscriminants')
plt.ylim([-0.1, 1.1])
plt.legend(loc='best')
plt.show()

w = np.hstack((eigen_pairs[0][1][:, np.newaxis].real,
               eigen_pairs[1][1][:, np.newaxis].real))
print('Matrix W:\n', w)

X_train_lda = X_train_std.dot(w)
colors = ['c', 'b', 'g']
markers=['s', 'x', 'o']
for l, c, m in zip(np.unique(y_train), colors, markers):
	plt.scatter(X_train_lda[y_train==l, 0] * (-1),
	            X_train_lda[y_train==l, 1] * (-1),
	            c=c, label=l, marker=m)

plt.xlabel('LD 1')
plt.ylabel('LD 2')
plt.legend(loc='lower right')
plt.show()
