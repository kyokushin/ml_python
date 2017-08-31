import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

class Perceptron(object):
    """パーセプトロンの分類器
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
        self.errors_ = []

        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi  # 重みw1...wmの更新
                self.w_[0] += update  # 重みwoの更新
                errors += int(update != 0.0)  # 重みの更新が0出ない場合は誤分類としてカウント

            self.errors_.append(errors)  # 反復回数ごとの誤差を格納

        return self

    def net_input(self, X):
        """総入力を計算"""
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        """1ステップ後のクラスラベルを返す"""
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
    plt.scatter(X[:50, 0], X[:50, 1], color='red', marker='o', label='setosa')
    plt.scatter(X[50:100, 0], X[50:100, 1], color='blue', marker='x', label='versicolor')
    plt.xlabel('sepal length [cm]')
    plt.ylabel('petal length [cm]')
    plt.legend(loc='upper left')
    plt.show()

    ppn = Perceptron(eta=0.1, n_iter=10)
    ppn.fit(X, y)
    plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Number of misclassifications')
    plt.show()


    plot_decision_regions(X, y, ppn)
    plt.xlabel('sepal length [cm]')
    plt.ylabel('petal length [cm]')
    plt.legend(loc="upper left")
    plt.show()

