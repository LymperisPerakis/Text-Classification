from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pickle
from sklearn.metrics import homogeneity_score, completeness_score
from sklearn.metrics import silhouette_score
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import TSNE
import seaborn as sns

# load documents for clustering
with open('train/cleaned_21.pickle', 'rb') as f:
    documents_list, label_list, labels, label_dummy, documents = pickle.load(f)


# define subclasses for better visualization
classes = len(labels)
classes = 6
small_doc = []
small_label = []
for i, document in enumerate(documents):
    if label_dummy[i] in range(classes):
        small_doc.append(document)
        small_label.append(label_dummy[i])

# transform documents and cluster with K-means
vec = TfidfVectorizer()
vec.fit(small_doc)
features = vec.transform(small_doc)
cls = MiniBatchKMeans(n_clusters=classes, random_state=0)
cls2 = KMeans(n_clusters=classes, random_state=0)
cls.fit(features)


cm = plt.cm.get_cmap('RdYlBu')
vmin=0
vmax=classes

# visualize clusters with PCA DR in 2D
pca = PCA(n_components=2, random_state=0)
reduced_features = pca.fit_transform(features.toarray())
reduced_cluster_centers = pca.transform(cls.cluster_centers_)
plt.scatter(reduced_features[:, 0], reduced_features[:, 1],
            c=cls.predict(features))
plt.scatter(reduced_cluster_centers[:, 0], reduced_cluster_centers[:, 1],
            marker='x', s=150, c='b', cmap =cm)
plt.title('Dimensionality Reduction using PCA')

# perform tSNE for all classes in 2D and 3D

tsne = TSNE(n_components=2).fit_transform(features.toarray(), label_dummy)
tsne3 = TSNE(n_components=3).fit_transform(features.toarray(), label_dummy)

# Visualize 2D
plt.figure(figsize=(16,10))
ax = sns.scatterplot(
    x=tsne[:,0], y=tsne[:,1],
    hue=label_dummy,
    palette=sns.color_palette("hls", 21),
    legend="full",
    alpha=0.3
)
ax.set_title('Dimensionality reduction using t-SNE')


# Visualize 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(tsne3[:, 0], tsne3[:, 1], tsne3[:, 2], c=label_dummy)
ax.set_title('Dimensionality reduction using t-SNE in 3D')


# DR using LSA
svd = TruncatedSVD(n_components=2, n_iter=7, random_state=0)
reduced_features = svd.fit_transform(features.toarray())
plt.scatter(reduced_features[:, 0], reduced_features[:, 1],
            c=cls.predict(features))

# DR using PCA in 3D
pca = PCA(n_components=3, random_state=0)
reduced_features = pca.fit_transform(features.toarray())
reduced_cluster_centers = pca.transform(cls.cluster_centers_)

# Plot 3D PCA DR
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(reduced_features[:, 0], reduced_features[:, 1], reduced_features[:, 2], c=cls.predict(features))

# Compute K means clustering performance
homogeneity_score(small_label, cls.predict(features))
silhouette_score(features, labels=cls.predict(features))
completeness_score(small_label, cls.predict(features))