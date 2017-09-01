import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import utils


class LogisticRegression(object):
	"""ADAptive LInear NEuron分類器
	パラメータ
	eta:float 学習率(0.0より大きく1.0以下の値)
	n_iter:int トレーニングデータのトレーニング数

	属性
	-----------
	w_:1次元配列 適合後の重み
	errors_:リスト 各エポックでの誤分類数
	"""

	def __init__(self, eta=0.01, n_iter=10):
		self.eta = eta
		self.n_iter = n_iter

	def fit(self, X, y):
		""" トレーニングデータに適合させる

		パラメータ
		__________
		X:{配列のようなデータ構造}, shape=[n_samples, n_features]
			トレーニングデータ
			n_sampleはサンプルの個数、n_featureは特徴量の個数
		y:配列のようなデータ構造、shape=[n_samples]
			目的変数

		戻り値
		___________
		self:object
		"""

		self.w_ = np.zeros(1 + X.shape[1])  # 1+m行のベクトル
		self.cost_ = []

		for _ in range(self.n_iter):
			# 活性化関数の出力の計算 wTx
			output = self.activation(X)
			# 誤差 yi - phi(zi)の計算
			errors = (y - output)
			# wm の更新
			# X.T.dot(errors)はコスト関数の勾配
			self.w_[1:] += self.eta * X.T.dot(errors)
			# w0 の更新
			self.w_[0] += self.eta * errors.sum()
			# コスト関数の計算 J(w) = 1/2 Sig[yi - phi(zi)]
			cost = self._logit_cost(y, output)

			self.cost_.append(cost)  # 反復回数ごとの誤差を格納

		return self

	def net_input(self, X):
		"""総入力関数"""
		return np.dot(X, self.w_[1:]) + self.w_[0]

	def _sigmoid(self, z):
		""" シグモイド関数"""
		return 1.0 / (1.0 + np.exp(-z))

	def _logit_cost(self, y, y_val):
		return -y.dot(np.log(y_val)) - ((1 - y).dot(np.log(1 - y_val)))

	def activation(self, X):
		z = self.net_input(X)
		return self._sigmoid(z)

	def predict(self, X):
		"""1ステップ後のクラスラベルを返す"""
		""" activation(X) >= 0.5 と同じ(sigmoid(0) = 0.5 """
		return np.where(self.net_input(X) >= 0.0, 1, -1)


def plot_decision_regions(X, y, classifier, resolution=0.02):
	# マーカーとカラーマップの準備
	markers = ('s', 'x', 'o', '^', 'v')
	colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
	cmap = ListedColormap(colors[:len(np.unique(y))])

	# 決定領域のプロット
	x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
	x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1

	# グリッドポイントの生成
	xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
	                       np.arange(x2_min, x2_max, resolution))
	Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
	Z = Z.reshape(xx1.shape)
	print(Z)

	plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)

	plt.xlim(xx1.min(), xx1.max())
	plt.ylim(xx2.min(), xx2.max())

	for idx, cl in enumerate(np.unique(y)):
		plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1], alpha=0.8, c=cmap(idx),
		            marker=markers[idx], label=cl)


if __name__ == '__main__':
	iris = utils.read_csv('iris.data', 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data')

	y = iris.iloc[0:100, 4].values
	y = np.where(y == 'Iris-setosa', 1, 0)


	X = iris.iloc[0:100, [0, 2]].values

	sc = StandardScaler()
	sc.fit(X)
	X_std = sc.transform(X)

	lr = LogisticRegression(n_iter=500, eta=0.2).fit(X_std, y)
	print(lr.predict(X_std))
	plt.plot(range(1, len(lr.cost_) + 1), np.log10(lr.cost_))
	plt.xlabel('Epochs')
	plt.ylabel('Cost')
	plt.title('Logistic Regression - Learning rate 0.2')
	plt.show()

	plot_decision_regions(X_std, y, classifier=lr, resolution=0.02)
	plt.title('Logistic Regression - Gradient Descent')
	plt.xlabel('sepal length [standardized]')
	plt.ylabel('petal length [standardized]')
	plt.legend(loc='upper left')
	plt.show()
