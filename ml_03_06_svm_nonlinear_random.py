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

    X_xor, y_xor = utils.generateDataSet()

    # gammaが小さいとトレーニングサンプルの影響力大。決定協会が滑らかに
    svm = SVC(kernel='rbf', gamma=0.10, C=10.0, random_state=0)
    svm.fit(X_xor, y_xor)
    utils.plot_decision_regions(X_xor, y_xor, classifier=svm)
    plt.xlabel('petal length [standardized')
    plt.ylabel('petal width [standardized')
    plt.legend(loc='upper left')
    plt.show()
