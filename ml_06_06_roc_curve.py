import utils
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from scipy import interp
from sklearn.metrics import roc_auc_score, accuracy_score

if __name__ == '__main__':
    df = utils.read_csv('wdbc.data',
                        'https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data')
    X = df.loc[:, 2:].values
    y = df.loc[:, 1].values

    le = LabelEncoder()
    y = le.fit_transform(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1)

    pipe_lr = Pipeline([('scl', StandardScaler()),
                        ('pca', PCA(n_components=2)),
                        ('clf', LogisticRegression(penalty='l2', random_state=0, C=100.0))
                        ])

    X_train2 = X_train[:, [4, 14]]

    cv = StratifiedKFold(n_splits=3, random_state=1).split(X_train2, y_train)
    fig = plt.figure(figsize=(7,5))
    mean_tpr = 0.0

    mean_fpr = np.linspace(0, 1, 100)
    all_tpr = []
    num_cv = 0
    for i, (train, test) in enumerate(cv):
        probas = pipe_lr.fit(X_train2[train], y_train[train]).predict_proba(X_train2[test]) # あるデータがクラスに属する確率の配列を返す
        print(probas)
        fpr, tpr, thresholds = roc_curve(y_train[test], probas[:, 1], pos_label=1) # label1のROC曲線
        mean_tpr += interp(mean_fpr, fpr, tpr) # 線形補完。fpr, tprのグラフに従って、mean_fprに対応するmean_tprを補完する。
        mean_tpr[0] = 0.0
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=1, label='ROC fold %d (area = %.2f' % (i+1, roc_auc))
        num_cv += 1

    plt.plot([0, 1], [0,1], linestyle='--', color=(0.6, 0.6, 0.6), label='random guessing')

    mean_tpr /= num_cv
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    plt.plot(mean_fpr, mean_tpr, 'k--', label='mean ROC (area = %0.2f)' % mean_auc, lw=2)
    plt.plot([0, 0, 1], [0, 1, 1], lw=2, linestyle=':', color='black', label='perfect performance')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('false positive rate')
    plt.ylabel('true positive rate')
    plt.title('Receiver Operator Characteristic')
    plt.legend(loc='lower right')
    plt.show()

    pipe_lr = pipe_lr.fit(X_train2, y_train)
    y_pred2 = pipe_lr.predict(X_test[:, [4, 14]])
    print('ROC AUC:%.3f' % roc_auc_score(y_true=y_test, y_score=y_pred2))
    print('Accuracy: %.2f' % accuracy_score(y_true=y_test, y_pred=y_pred2))