from scipy.spatial.distance import pdist, squareform
from scipy import exp
from scipy.linalg import eigh
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from matplotlib.ticker import FormatStrFormatter
from sklearn.datasets import make_moons
from sklearn.datasets import make_circles


def rbf_kernel_pca(X, gamma, n_components):
    sq_dists = pdist(X, 'sqeuclidean')
    mat_sq_dists = squareform(sq_dists)  # 距離ベクトルを距離行列に変換

    K = exp(-gamma * mat_sq_dists)

    N = K.shape[0]
    one_n = np.ones((N, N)) / N
    K = K - one_n.dot(K) - K.dot(one_n) + one_n.dot(K).dot(one_n)

    eigvals, eigvecs = eigh(K)

    alphas = np.column_stack((eigvecs[:, -i] for i in range(1, n_components + 1)))
    lambdas = [eigvals[-i] for i in range(1, n_components + 1)]

    return alphas, lambdas


def project_x(x_new, X, gamma, alphas, lambdas):
    pair_dist = np.array([np.sum((x_new - row) ** 2) for row in X])
    k = np.exp(-gamma * pair_dist)
    return k.dot(alphas / lambdas)


X, y = make_moons(n_samples=100, random_state=123)
#X, y = make_circles(n_samples=1000, random_state=123, noise=0.1, factor=0.2)
n = int(X.shape[0])
print(n)
plt.scatter(X[y == 0, 0], X[y == 0, 1], color='red', marker='^', alpha=0.5)
plt.scatter(X[y == 1, 0], X[y == 1, 1], color='blue', marker='o', alpha=0.5)
plt.show()

scikit_pca = PCA(n_components=2)
X_spca = scikit_pca.fit_transform(X)

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(7, 3))

ax[0].scatter(X_spca[y == 0, 0], X_spca[y == 0, 1], color='red', marker='^', alpha=0.5)
ax[0].scatter(X_spca[y == 1, 0], X_spca[y == 1, 1], color='blue', marker='o', alpha=0.5)

ax[1].scatter(X_spca[y == 0, 0], np.zeros((n // 2, 1)) + 0.02, color='red', marker='^', alpha=0.5)
ax[1].scatter(X_spca[y == 1, 0], np.zeros((n // 2, 1)) - 0.02, color='blue', marker='o', alpha=0.5)

ax[0].set_xlabel('PC1')
ax[0].set_ylabel('PC2')
ax[1].set_ylim([-1, 1])
ax[1].set_yticks([])
ax[1].set_xlabel('PC1')
plt.show()

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(7, 3))
alphas, lambdas = rbf_kernel_pca(X, gamma=15, n_components=2)

ax[0].scatter(alphas[y == 0, 0], alphas[y == 0, 1], color='red', marker='^', alpha=0.5)
ax[0].scatter(alphas[y == 1, 0], alphas[y == 1, 1], color='blue', marker='o', alpha=0.5)

ax[1].scatter(alphas[y == 0, 0], np.zeros((n // 2, 1)) + 0.02, color='red', marker='^', alpha=0.5)
ax[1].scatter(alphas[y == 1, 0], np.zeros((n // 2, 1)) - 0.02, color='blue', marker='o', alpha=0.5)

ax[0].set_xlabel('PC1')
ax[0].set_ylabel('PC2')
ax[1].set_ylim([-1, 1])
ax[1].set_yticks([])
ax[1].set_xlabel('PC1')

ax[0].xaxis.set_major_formatter(FormatStrFormatter('%0.1f'))
ax[1].xaxis.set_major_formatter(FormatStrFormatter('%0.1f'))

plt.show()


alphas, lambdas = rbf_kernel_pca(X, gamma=15, n_components=1)
x_new = X[25]
x_reproj = project_x(x_new, X, gamma=15, alphas=alphas, lambdas=lambdas)
print('x_proj, x_reproj:', alphas[25], ',', x_reproj)

