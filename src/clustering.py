from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pickle
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import homogeneity_score, completeness_score
from sklearn.metrics import silhouette_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import (KNeighborsClassifier,
                               NeighborhoodComponentsAnalysis)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD

# https://sanjayasubedi.com.np/nlp/nlp-with-python-document-clustering/
with open('training_data/cleaned_docs_25_classes.pickle', 'rb') as f:
    documents_list, label_list, labels, label_dummy, documents = pickle.load(f)

classes = 6
small_doc = []
small_label = []
for i, document in enumerate(documents):
    if label_dummy[i] in range(2, classes):
        small_doc.append(document)
        small_label.append(label_dummy[i])

vec = TfidfVectorizer()
vec.fit(small_doc)
features = vec.transform(small_doc)
cls = MiniBatchKMeans(n_clusters=classes-2, random_state=0)
cls.fit(features)

#2D
pca = PCA(n_components=2, random_state=0)
reduced_features = pca.fit_transform(features.toarray())
reduced_cluster_centers = pca.transform(cls.cluster_centers_)
plt.scatter(reduced_features[:, 0], reduced_features[:, 1], c=cls.predict(features))
plt.scatter(reduced_cluster_centers[:, 0], reduced_cluster_centers[:, 1], marker='x', s=150, c='b')

# LSA
svd = TruncatedSVD(n_components=2, n_iter=7, random_state=0)
reduced_features = svd.fit_transform(features.toarray())
plt.scatter(reduced_features[:, 0], reduced_features[:, 1], c=cls.predict(features))

#3D
pca = PCA(n_components=3, random_state=0)
reduced_features = pca.fit_transform(features.toarray())
reduced_cluster_centers = pca.transform(cls.cluster_centers_)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(reduced_features[:, 0], reduced_features[:, 1], reduced_features[:,2], c=cls.predict(features))

homogeneity_score(small_label, cls.predict(features))
silhouette_score(features, labels=cls.predict(features))
completeness_score(small_label, cls.predict(features))

# lda after knn
n_neighbors = 3
random_state = 0
X_train, X_test, y_train, y_test = \
    train_test_split(features, small_label, test_size=0.5, stratify=small_label,
                     random_state=random_state)
dim = len(small_doc[0])
n_classes = len(np.unique(small_label))

pca = make_pipeline(StandardScaler(),
                    PCA(n_components=2, random_state=random_state))

lda = make_pipeline(StandardScaler(),
                    LinearDiscriminantAnalysis(n_components=2))

nca = make_pipeline(StandardScaler(),
                    NeighborhoodComponentsAnalysis(n_components=2,
                                                   random_state=random_state))
# Use a nearest neighbor classifier to evaluate the methods
knn = KNeighborsClassifier(n_neighbors=n_neighbors)

# Make a list of the methods to be compared
dim_reduction_methods = [('PCA', pca), ('LDA', lda), ('NCA', nca)]

# plt.figure()
for i, (name, model) in enumerate(dim_reduction_methods):
    plt.figure()
    # plt.subplot(1, 3, i + 1, aspect=1)

    # Fit the method's model
    model.fit(X_train, y_train)

    # Fit a nearest neighbor classifier on the embedded training set
    knn.fit(model.transform(X_train), y_train)

    # Compute the nearest neighbor accuracy on the embedded test set
    acc_knn = knn.score(model.transform(X_test), y_test)

    # Embed the data set in 2 dimensions using the fitted model
    X_embedded = model.transform(features)

    # Plot the projected points and show the evaluation score
    plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=small_label, s=30, cmap='Set1')
    plt.title("{}, KNN (k={})\nTest accuracy = {:.2f}".format(name,
                                                              n_neighbors,
                                                              acc_knn))
plt.show()