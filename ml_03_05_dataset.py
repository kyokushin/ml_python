import utils
import matplotlib.pyplot as plt

if __name__ == '__main__':
    X_xor, y_xor = utils.generateDataSet()

    plt.scatter(X_xor[y_xor == 1, 0], X_xor[y_xor == 1, 1], c='b', marker='x', label='1')
    plt.scatter(X_xor[y_xor == -1, 0], X_xor[y_xor == -1, 1], c='r', marker='s', label='-1')

    plt.xlim([-3, 3])
    plt.ylim([-3, 3])
    plt.legend(loc='best')
    plt.show()
