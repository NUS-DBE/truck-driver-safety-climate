from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
from sklearn.cluster import HDBSCAN,BisectingKMeans
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.inspection import PartialDependenceDisplay, permutation_importance
from sklearn.cluster import DBSCAN, Birch,AgglomerativeClustering







import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.cm as cm


# Load the provided data
data = pd.read_csv('concatenated_data.csv')

# Extracting features and labels
X = data.iloc[:, :-1].values
voted_labels = data.iloc[:, -1].values





from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import MDS, Isomap, LocallyLinearEmbedding, SpectralEmbedding
# from umap import UMAP
# Standardize the data
scaler = StandardScaler()
X_standardized = scaler.fit_transform(X)

# PCA with standardized data
pca_standardized = PCA(n_components=2,whiten=True)
X_pca_standardized = pca_standardized.fit_transform(X_standardized)

# LLE with different n_neighbors values
lle_results = {}
n_neighbors_values = [ 50,100,150,200]
for n in n_neighbors_values:
    lle = LocallyLinearEmbedding(n_components=2, n_neighbors=n, method='standard')
    lle_results[n] = lle.fit_transform(X_standardized)

# Plotting the results
plt.figure(figsize=(18, 10))

# PCA plot
plt.subplot(2, 3, 1)
plt.scatter(X_pca_standardized[:, 0], X_pca_standardized[:, 1], c=voted_labels, cmap=cm.jet)
plt.title('PCA (Standardized)')
plt.colorbar()

# LLE plots
for i, (n, reduced_data) in enumerate(lle_results.items(), 2):
    plt.subplot(2, 3, i)
    plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=voted_labels, cmap=cm.jet)
    plt.title(f'LLE (n_neighbors={n})')
    plt.colorbar()

plt.tight_layout()
plt.show()
exit(0)
# Perform t-SNE dimensionality reduction
tsne = TSNE()
X_tsne = tsne.fit_transform(X)

# Plot t-SNE results
plt.figure(figsize=(10, 8))
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=voted_labels, cmap=cm.jet)
plt.title('t-SNE Scatter Plot')
plt.colorbar()
plt.show()
exit(0)




df = pd.read_csv('trucking factors only (wo TSF).csv')
X = df.copy()[['OSC1','OSC2','OSC3','GSC1','GSC2','GSC3']]

m = KMeans(n_clusters = 2, init = 'k-means++', random_state=0)
m.fit(X)
labels = m.labels_-1
Y1=labels
# print(sum(Y))


m = DBSCAN(eps=0.946, min_samples=230)
m.fit(X)
labels = m.labels_
Y2 = [-1 if x == 0 else 0 for x in labels]
# print(sum(Y))



m = AgglomerativeClustering(n_clusters=2)
m.fit(X)
labels = m.labels_-1
Y3=labels
# print(sum(Y))

from sklearn.cluster import MeanShift, estimate_bandwidth

df = pd.read_csv('trucking factors only (wo TSF).csv')
X = df.copy()[['OSC1','OSC2','OSC3','GSC1','GSC2','GSC3']]
bandwidth = estimate_bandwidth(X, quantile=0.25)
m = MeanShift(bandwidth=bandwidth)
m.fit(X)

ms_labels = m.fit_predict(X)-1
Y4=ms_labels
# print(sum(Y))


m = Birch(n_clusters=2)
m.fit(X)
labels = m.labels_-1
Y5=labels
# print(sum(Y))



combined_labels =np.array(list( zip(Y1, Y2,Y3,Y4,Y5)))

voted_labels = np.where(np.sum(combined_labels == -1, axis=1) >= 3, -1, 0).reshape(-1, 1)
# voted_labels.shape, np.unique(voted_labels, return_counts=True)
print(voted_labels.shape, np.unique(voted_labels, return_counts=True))
final=np.hstack((X, voted_labels))

csv_path = "concatenated_data.csv"
np.savetxt(csv_path, final, delimiter=",", fmt="%d")

