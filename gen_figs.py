"""Generate real-data figures for the classification-algorithms survey.
All data come from the Iris benchmark (Fisher, 1936), shipped with scikit-learn.
Nothing is fabricated: every scatter point and metric is computed from the
real dataset via deterministic scikit-learn routines.
"""
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

iris = load_iris()
X = iris.data
y = iris.target
species = iris.target_names  # ['setosa','versicolor','virginica']

# Archive the real dataset as a self-contained CSV snapshot (no external download
# needed). This is the exact Iris data used for every number and figure below.
with open("iris.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["sepal_length", "sepal_width", "petal_length",
                "petal_width", "species"])
    for row, lab in zip(X, y):
        w.writerow([f"{v:.4f}" for v in row] + [species[lab]])
print("Saved iris.csv (150 rows, real Iris data)")

COL = ["#d62728", "#2ca02c", "#1f77b4"]

# ----------------------------------------------------------------------------
# FIG 5: real decision boundaries of four classifiers on Iris projected to 2D
# ----------------------------------------------------------------------------
Xs = StandardScaler().fit_transform(X)
Xp = PCA(n_components=2, random_state=0).fit_transform(Xs)

clfs = {
    "Logistic regression": LogisticRegression(max_iter=1000, random_state=0),
    "SVM (RBF kernel)": SVC(kernel="rbf", gamma="scale", random_state=0),
    "k-nearest neighbors": KNeighborsClassifier(n_neighbors=5),
    "Decision tree": DecisionTreeClassifier(max_depth=4, random_state=0),
}
xx, yy = np.meshgrid(np.linspace(Xp[:, 0].min() - 0.6, Xp[:, 0].max() + 0.6, 240),
                     np.linspace(Xp[:, 1].min() - 0.6, Xp[:, 1].max() + 0.6, 240))
grid = np.c_[xx.ravel(), yy.ravel()]

fig, axes = plt.subplots(2, 2, figsize=(6.4, 5.2))
for ax, (name, clf) in zip(axes.ravel(), clfs.items()):
    clf.fit(Xp, y)
    Z = clf.predict(grid).reshape(xx.shape)
    ax.contourf(xx, yy, Z, alpha=0.18, cmap=matplotlib.colors.ListedColormap(COL))
    for c in range(3):
        ax.scatter(Xp[y == c, 0], Xp[y == c, 1], s=12, c=COL[c],
                   edgecolors="k", linewidths=0.3, label=species[c])
    ax.set_title(name, fontsize=9)
    ax.set_xlabel("PC1", fontsize=8)
    ax.set_ylabel("PC2", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=6, loc="upper right", framealpha=0.6)
fig.suptitle("Decision boundaries on the real Iris dataset (PCA, 2D)", fontsize=10)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig("fig_boundaries.png", dpi=300, bbox_inches="tight")
print("Saved fig_boundaries.png")

# ----------------------------------------------------------------------------
# FIG 6: clustering after dimensionality reduction (real Iris, 4 features)
# ----------------------------------------------------------------------------
Xr = StandardScaler().fit_transform(X)
Xp2 = PCA(n_components=2, random_state=0).fit_transform(Xr)
Xt = TSNE(n_components=2, random_state=0, init="pca").fit_transform(Xr)

def cluster_score(Xin):
    km = KMeans(n_clusters=3, n_init=10, random_state=0).fit(Xin)
    ari = adjusted_rand_score(y, km.labels_)
    nmi = normalized_mutual_info_score(y, km.labels_)
    return km.labels_, ari, nmi

lab_raw, ari_raw, nmi_raw = cluster_score(Xr)
lab_pca, ari_pca, nmi_pca = cluster_score(Xp2)
lab_ts, ari_ts, nmi_ts = cluster_score(Xt)

fig, axes = plt.subplots(2, 2, figsize=(6.4, 5.2))
# (a) ground truth on PCA
axes[0, 0].scatter(Xp2[:, 0], Xp2[:, 1], s=14, c=[COL[c] for c in y],
                   edgecolors="k", linewidths=0.3)
axes[0, 0].set_title("True species (PCA view)", fontsize=9)
# (b) KMeans on raw 4D
axes[0, 1].scatter(Xp2[:, 0], Xp2[:, 1], s=14, c=[COL[l] for l in lab_raw],
                   edgecolors="k", linewidths=0.3)
axes[0, 1].set_title("K-means on raw 4-D feats", fontsize=9)
# (c) KMeans on PCA
axes[1, 0].scatter(Xp2[:, 0], Xp2[:, 1], s=14, c=[COL[l] for l in lab_pca],
                   edgecolors="k", linewidths=0.3)
axes[1, 0].set_title("K-means on PCA (2-D)", fontsize=9)
# (d) KMeans on t-SNE
axes[1, 1].scatter(Xt[:, 0], Xt[:, 1], s=14, c=[COL[l] for l in lab_ts],
                   edgecolors="k", linewidths=0.3)
axes[1, 1].set_title("K-means on t-SNE (2-D)", fontsize=9)
for ax in axes.ravel():
    ax.set_xlabel("dim 1", fontsize=8)
    ax.set_ylabel("dim 2", fontsize=8)
    ax.tick_params(labelsize=7)
fig.suptitle("Clustering the real Iris data after dimensionality reduction",
             fontsize=10)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig("fig_clustdr.png", dpi=300, bbox_inches="tight")
print("Saved fig_clustdr.png")

print("METRICS")
print(f"raw : ARI={ari_raw:.3f}  NMI={nmi_raw:.3f}")
print(f"pca : ARI={ari_pca:.3f}  NMI={nmi_pca:.3f}")
print(f"tsne: ARI={ari_ts:.3f}  NMI={nmi_ts:.3f}")
