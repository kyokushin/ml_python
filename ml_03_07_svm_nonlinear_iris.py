import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.svm import SVC
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import utils

if __name__ == '__main__':


    iris = datasets.load_iris()

    X = iris.data[:, [2, 3]]
    y = iris.target

    print("Class labels:", np.unique(y))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

    sc = StandardScaler()
    sc.fit(X_train)
    X_train_std = sc.transform(X_train)
    X_test_std = sc.transform(X_test)

    X_combined_std = np.vstack((X_train_std, X_test_std))
    y_combined = np.hstack((y_train, y_test))

    # gammaが小さいとトレーニングサンプルの影響力大。決定協会が滑らかに
    #svm = SVC(kernel='rbf', gamma=0.20, C=1.0, random_state=0)
    svm = SVC(kernel='rbf', gamma=100, C=1.0, random_state=0)
    svm.fit(X_train_std, y_train)
    utils.plot_decision_regions(X_combined_std, y_combined, classifier=svm, test_idx=range(105, 150))
    plt.xlabel('petal length [standardized')
    plt.ylabel('petal width [standardized')
    plt.legend(loc='upper left')
    plt.show()
