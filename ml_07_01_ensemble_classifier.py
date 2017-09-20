from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.externals import six
from sklearn.base import clone
from sklearn.pipeline import _name_estimators
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
import matplotlib.pyplot as plt
from itertools import product
from sklearn.model_selection import GridSearchCV

class MajorityVoteClassifier(BaseEstimator, ClassifierMixin):
    """
    多数決アンサンブル分類機

    パラメータ
    -------
    classifiers : array-like, shape = [n_classifier]
        アンサンブルで利用する分類機

    vote : str, {'classlabel', 'probability'} (default: 'classlabel')
        'classlabel' の場合、クラスラベルの予測はクラスラベルのargmaxに基づく
        'probability'の場合、クラスラベルの予測はクラスラベルの所属確率のargmaxに基づく
        （分類器が町政済みであることが推奨される）

    weights: array-like, shape = [n_classifiers] (optional, default=None)
        'int'または'float'型の値のリストが提供された場合、分類器は重要度で重みづけされる
        'weights=None'の場合は均一な重みを使用
    """

    def __init__(self, classifiers, vote='classlabel', weights=None):
        self.classifiers = classifiers
        self.named_classifiers = {key: value for key, value in _name_estimators(classifiers)}
        self.vote = vote
        self.weights = weights

    def fit(self, X, y):
        """
        分類器を学習させる
        :param X: {array-like, sparse matrix},
            shape = [n_samples, n_features]
            トレーニングサンプルからなる行列
        :param y: array-like, shape = [n_camples]
            クラスラベルのベクトル
        :return: self : object
        """
        # LabelEncoderを使ってクラスラベルが0から始まるようにエンコードする
        # self.predictのnp.argmax呼び出しで重要となる

        self.lablenc_ = LabelEncoder()
        self.lablenc_.fit(y)
        self.classes_ = self.lablenc_.classes_
        self.classifiers_ = []
        for clf in self.classifiers:
            fitted_clf = clone(clf).fit(X, self.lablenc_.transform(y))
            self.classifiers_.append(fitted_clf)

        return self

    def predict(self, X):
        """
            Xのクラスラベルを予測する
        :param X: {array-like, sparse matrix}, shape = [n_samples, n_features]
            トレーニングサンプルからなる行列
        :return: maj_vote : array-like, shape = [n_samples]
            予測されたクラスラベル
        """
        if self.vote == 'probability':
            maj_vote = np.argmax(self.predict_proba(X), axis=1)

        else:  # 'classlabel'での多数決
            # clf.predict呼び出し結果を収集
            predictions = np.asarray([clf.predict(X) for clf in self.classifiers_]).T
            # 各サンプルのクラス確率に重みを掛けて足し合わせた値が最大となる列番号を配列として返す
            maj_vote = np.apply_along_axis(
                lambda x: np.argmax(np.bincount(x, weights=self.weights)),
                axis=1, arr=predictions)

        # 各サンプルに確率の最大値を与えるクラスラベルを抽出
        maj_vote = self.lablenc_.inverse_transform(maj_vote)
        return maj_vote

    def predict_proba(self, X):
        """
        Xのクラス確率を予測する

        :param X: {array-like, sparse matrix}, shape = [n_samples, n_features]
            トレーニングベクトル: n_samplesはサンプルの個数、n_featuresは特徴量の個数
        :return: avg_proba : array_like, shape = [n_samples, n_classes]
            各サンプルに対する各クラスで重みづけた平均確率
        """

        probas = np.asarray([clf.predict_proba(X) for clf in self.classifiers_])
        avg_proba = np.average(probas, axis=0, weights=self.weights)
        return avg_proba

    def get_params(self, deep=True):
        """
        GridSearchの実行時に分類器のパラメータ名を取得
        :param deep:
        :return:
        """

        if not deep:
            return super(MajorityVoteClassifier, self).get_params(deep=False)

        else:
            # キーを"分類器の名前__パラメータ名",
            # バリューをパラメータの値とするディクショナリを生成
            out = self.named_classifiers.copy()
            for name, step in six.iteritems(self.named_classifiers):
                for key, value in six.iteritems(step.get_params(deep=True)):
                    out['%s__%s' % (name, key)] = value

            return out


if __name__ == '__main__':
    iris = datasets.load_iris()
    X, y = iris.data[50:, [1, 2]], iris.target[50:]
    le = LabelEncoder()
    y = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=1)

    clf1 = LogisticRegression(penalty='l2', C=0.001, random_state=0)
    clf2 = DecisionTreeClassifier(max_depth=1, criterion='entropy', random_state=0)
    clf3 = KNeighborsClassifier(n_neighbors=1, p=2, metric='minkowski')
    pipe1 = Pipeline([['sc', StandardScaler()], ['clf', clf1]])
    pipe3 = Pipeline([['sc', StandardScaler()], ['clf', clf3]])

    mv_clf = MajorityVoteClassifier(classifiers=[pipe1, clf2, pipe3])
    clf_labels = ['Logistic Regression', 'Decision Tree', 'KNN', 'MajorityVoting']
    all_clf = [pipe1, clf2, pipe3, mv_clf]
    print('10-fold cross validation:\n')
    for clf, label in zip(all_clf, clf_labels):
        scores = cross_val_score(estimator=clf, X=X_train, y=y_train, cv=10, scoring='roc_auc')
        print("ROC AUC: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))

    colors = ['black', 'orange', 'blue', 'green']
    linestyles = [':', '--', '-.', '-']
    for clf, label, clr, ls in zip(all_clf, clf_labels, colors, linestyles):
        y_pred = clf.fit(X_train, y_train).predict_proba(X_test)[:, 1]
        fpr, tpr, thresholds = roc_curve(y_true=y_test, y_score=y_pred)
        roc_auc = auc(x=fpr, y=tpr)
        plt.plot(fpr, tpr, color=clr, linestyle=ls, label='%s (auc = %0.2f)' % (label, roc_auc))

    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', linewidth=2)
    plt.xlim([-0.1, 1.1])
    plt.ylim([-0.1, 1.1])
    plt.grid()
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.show()

    sc = StandardScaler()
    X_train_std = sc.fit_transform(X_train)
    x_min = X_train_std[:, 0].min() - 1
    x_max = X_train_std[:, 0].max() + 1
    y_min = X_train_std[:, 1].min() - 1
    y_max = X_train_std[:, 1].max() + 1

    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
    f, axarr = plt.subplots(nrows=2, ncols=2, sharex='col', sharey='row', figsize=(7, 5))
    for idx, clf, tt, in zip(product([0, 1], [0, 1]), all_clf, clf_labels):
        clf.fit(X_train_std, y_train)
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        axarr[idx[0], idx[1]].contourf(xx, yy, Z, alpha=0.3)
        axarr[idx[0], idx[1]].scatter(X_train_std[y_train == 0, 0], X_train_std[y_train == 0, 1],
                                      c='blue', marker='^', s=50)
        axarr[idx[0], idx[1]].scatter(X_train_std[y_train == 1, 0], X_train_std[y_train == 1, 1],
                                      c='red', marker='o', s=50)
        axarr[idx[0], idx[1]].set_title(tt)

    plt.text(-3.5, -4.5, s='Sepal width [standardized]', ha='center', va='center', fontsize=12)
    plt.text(-10.5, 4.5, s='Petal length [standardized]', ha='center', va='center', fontsize=12, rotation=90)
    plt.show()

    params = {'decisiontreeclassifier__max_depth': [1, 2], 'pipeline-1__clf__C': [0.001, 0.1, 100.0]}
    grid = GridSearchCV(estimator=mv_clf, param_grid=params, cv=10, scoring='roc_auc')
    grid.fit(X_train, y_train)

    print('grid scored:', grid.grid_scores_)
    print('cv results:', grid.cv_results_['mean_test_score'])

    for params, mean_score, scores in zip(grid.cv_results_['params'], grid.cv_results_['mean_test_score'], grid.cv_results_['std_test_score']):
        print('%0.3f+/-%0.2f %r' % (mean_score, scores.std()/2, params))

    print('Best parameters: %s' % grid.best_params_)
    print('Accuracy: %.2f' % grid.best_score_)