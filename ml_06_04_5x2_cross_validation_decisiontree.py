import utils
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

if __name__ == '__main__':
    df = utils.read_csv('wdbc.data',
                        'https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data')
    X = df.loc[:, 2:].values
    y = df.loc[:, 1].values

    le = LabelEncoder()
    y = le.fit_transform(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1)

    gs = GridSearchCV(estimator=DecisionTreeClassifier(), param_grid=[{'max_depth': [1, 2, 3, 4, 5, 6, 7, None]}],
                      scoring='accuracy', cv=2)
    scores = cross_val_score(gs, X_train, y_train, scoring='accuracy', cv=5)
    print('CV accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))
