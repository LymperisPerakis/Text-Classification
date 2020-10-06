import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.decomposition import NMF
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
from sklearn.utils import parallel_backend
from joblib import Memory
from shutil import rmtree

# define memory for reusing some calculations
location = 'cachedir'
memory = Memory(location=location, verbose=0)

# define pipeline
pipe = Pipeline([
    ('reduce_dim', 'passthrough'),
    ('classify', SGDClassifier()),
], memory=memory)

# define parameter grid
N_FEATURES_OPTIONS = [10, 100, 500, 1000, 2000]
C_OPTIONS = [1, 10, 100, 1000]
param_grid = [
    {
        'reduce_dim': [TruncatedSVD(n_iter=7, random_state=0), NMF(max_iter=400)],
        'reduce_dim__n_components': N_FEATURES_OPTIONS,
        'classify__C': C_OPTIONS
    },
    {
        'reduce_dim': [SelectKBest(chi2)],
        'reduce_dim__k': N_FEATURES_OPTIONS,
        'classify__C': C_OPTIONS
    },
]

# labels of the DR methods
reducer_labels = ['LSA', 'NMF', 'KBest(chi2)']

# define the grid search
grid = GridSearchCV(pipe, n_jobs=4, param_grid=param_grid)

# load the documents to be used for training
with open('train/cleaned_21.pickle', 'rb') as f:
    documents_list, label_list, labels, label_dummy, documents = pickle.load(f)

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(documents)

# train the models with grid search using multithreading
with parallel_backend('multiprocessing'):
    grid.fit(X, label_dummy)

# save the fitted grid
with open('train/dim_red_grid_20_09.pickle', 'wb') as f:
    pickle.dump(grid, f)

#clear memory
memory.clear(warn=False)
rmtree(location)

# Calculate the best score for each DR method and feature
mean_scores = np.array(grid.cv_results_['mean_test_score'])
mean_scores = mean_scores.reshape(len(C_OPTIONS), -1, len(N_FEATURES_OPTIONS))
mean_scores = mean_scores.max(axis=0)
bar_offsets = (np.arange(len(N_FEATURES_OPTIONS)) *
               (len(reducer_labels) + 1) + .5)

# plots the features and scores of the DR methods
plt.figure()
COLORS = 'bgrcmyk'
for i, (label, reducer_scores) in enumerate(zip(reducer_labels, mean_scores)):
    plt.bar(bar_offsets + i, reducer_scores, label=label, color=COLORS[i])

plt.title("Comparing Feature Reduction techniques")
plt.xlabel('Reduced Number of Features')
plt.xticks(bar_offsets + len(reducer_labels) / 2, N_FEATURES_OPTIONS)
plt.ylabel('Hardware Classification Accuracy')
plt.ylim((0, 1))
plt.legend(loc='upper left')
plt.show()
