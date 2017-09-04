import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.ensemble import RandomForestClassifier
from sklearn import datasets
from sklearn.model_selection import train_test_split
import utils

if __name__ == '__main__':


    iris = datasets.load_iris()

    X = iris.data[:, [2, 3]]
    y = iris.target

    print("Class labels:", np.unique(y))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)


    X_combined = np.vstack((X_train, X_test))
    y_combined = np.hstack((y_train, y_test))

    forest = RandomForestClassifier(criterion='entropy', n_estimators=10, random_state=1, n_jobs=2)
    forest.fit(X_train, y_train)

    utils.plot_decision_regions(X_combined, y_combined, classifier=forest, test_idx=range(105, 150))
    plt.xlabel('petal length [standardized')
    plt.ylabel('petal width [standardized')
    plt.legend(loc='upper left')
    plt.show()
