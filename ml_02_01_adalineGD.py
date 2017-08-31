import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


class AdalineGD(object):
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
            output = self.net_inp78ut(X)
            # 誤差 yi - phi(zi)の計算
            errors = (y - output)
            # wm の更新
            # X.T.dot(errors)はコスト関数の勾配
            self.w_[1:] += self.eta * X.T.dot(errors)
            # w0 の更新
            self.w_[0] += self.eta * errors.sum()
            # コスト関数の計算 J(w) = 1/2 Sig[yi - phi(zi)]
            cost = (errors ** 2).sum() / 2.0

            self.cost_.append(cost)  # 反復回数ごとの誤差を格納

        return self

    def net_input(self, X):
        """総入力を計算"""
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def activation(self, X):
        return self.net_input(X)

    def predict(self, X):
        """1ステップ後のクラスラベルを返す"""
        return np.where(self.activation(X) >= 0.0, 1, -1)


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
    print(xx1)
    print(xx2)
    print([xx1.ravel(), xx2.ravel()])
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    print(Z)
    Z = Z.reshape(xx1.shape)
    print(Z)

    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)

    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1], alpha=0.8, c=cmap(idx),
                    marker=markers[idx], label=cl)


if __name__ == '__main__':
    df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)

    y = df.iloc[0:100, 4].values
    y = np.where(y == 'Iris-setosa', -1, 1)
    print(y)
    X = df.iloc[0:100, [0, 2]].values

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))
    ada1 = AdalineGD(n_iter=10, eta=0.01).fit(X, y)

    ax[0].plot(range(1, len(ada1.cost_) + 1), np.log10(ada1.cost_), marker='o')
    ax[0].set_xlabel('Epochs')
    ax[0].set_ylabel('log(Sum-squared-error)')
    ax[0].set_title('Adaline - Learning rate 0.01')

    ada2 = AdalineGD(n_iter=10, eta=0.0001).fit(X, y)
    ax[1].plot(range(1, len(ada2.cost_) + 1), ada2.cost_, marker='o')
    ax[1].set_xlabel('Epochs')
    ax[1].set_ylabel('Sum-squared-error')
    ax[1].set_title('Adaline - Learning rate 0.0001')
    plt.show()

    X_std = np.copy(X)
    X_std[:, 0] = (X[:, 0] - X[:, 0].mean()) / X[:, 0].std()
    X_std[:, 1] = (X[:, 1] - X[:, 1].mean()) / X[:, 1].std()

    ada = AdalineGD(n_iter=15, eta=0.01)
    ada.fit(X_std, y)
    plot_decision_regions(X_std, y, classifier=ada)
    plt.title('Adaline - Gradient Descent')
    plt.xlabel('sepal length [standardized]')
    plt.ylabel('petal length [standardized]')
    plt.legend(loc='upper left')
    plt.show()

    plt.plot(range(1, len(ada.cost_) + 1), ada.cost_, marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Sum-squared-error')
    plt.show()

