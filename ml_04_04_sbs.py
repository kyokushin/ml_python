import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.base import clone
from itertools import combinations
from sklearn.metrics import accuracy_score
import utils


class SBS():
	"""
	逐次後退選択を実行するクラス
	"""

	def __init__(self, estimator, k_features,
	             scoring=accuracy_score, test_size=0.25,
	             random_state=1):
		self.scoring = scoring  # 特徴量を選択する指標
		self.estimator = clone(estimator)  # 推定器
		self.k_features = k_features  # 選択する特徴量の個数
		self.test_size = test_size  # テストデータの割合
		self.random_state = random_state  # 乱数のシード

	def fit(self, X, y):
		X_train, X_test, y_train, y_test = train_test_split(
			X, y, test_size=self.test_size,
			random_state=self.random_state
		)
		dim = X_train.shape[1]  # 次元数
		self.indices_ = tuple(range(dim))
		self.subsets_ = [self.indices_]  # 部分集合。最初は全次元
		# 初期のスコア
		score = self._calc_score(X_train, y_train,
		                         X_test, y_test,
		                         self.indices_)
		self.scores_ = [score]  # 初期のスコアを保存
		while dim > self.k_features:
			scores = []
			subsets = []

			# p:1次元すくない特徴量の組み合わせ
			for p in combinations(self.indices_, r=dim - 1):
				score = self._calc_score(X_train, y_train,
				                         X_test, y_test, p)
				scores.append(score)
				subsets.append(p)

			best = np.argmax(scores)
			self.indices_ = subsets[best]
			self.subsets_.append(self.indices_)
			dim -= 1

			self.scores_.append(scores[best])
		self.k_score_ = self.scores_[-1]

		return self

	# 特徴選択
	def transform(self, X):
		return X[:, self.indices_]

	def _calc_score(self, X_train, y_train, X_test, y_test,
	                indices):
		self.estimator.fit(X_train[:, indices], y_train)
		y_pred = self.estimator.predict(X_test[:, indices])
		score = self.scoring(y_test, y_pred)
		return score


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

knn = KNeighborsClassifier(n_neighbors=2)
sbs = SBS(knn, k_features=1)
sbs.fit(X_train_std, y_train)

k_feat = [len(k) for k in sbs.subsets_]

plt.plot(k_feat, sbs.scores_, marker='o')
plt.ylim([0.7, 1.1])
plt.ylabel('Accuracy')
plt.xlabel('Number of features')
plt.grid()
plt.show()

k5 = list(sbs.subsets_[8])
print(df_wine.columns[1:][k5])

knn.fit(X_train_std, y_train)
print('# No selected feature')
print('Training accuracy:', knn.score(X_train_std, y_train))
print('Test accuracy', knn.score(X_test_std, y_test))

knn.fit(X_train_std[:, k5], y_train)
print('# Selected feature')
print('Training accuracy:', knn.score(X_train_std[:, k5], y_train))
print('Test accuracy', knn.score(X_test_std[:, k5], y_test))
