import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from nltk.tokenize import RegexpTokenizer
from sklearn.feature_extraction.text import CountVectorizer
from yellowbrick.text import FreqDistVisualizer

from src.classification.train_classification_model import \
    TrainClassificationModel

# load documents
train_model = TrainClassificationModel(model='fasttext', process=False)
documents_list, label_list, labels, label_dummy, documents = \
    train_model.preprocess_files()
X_train, X_test, y_train, y_test = train_model.split_training_files()

# visualize the words with
vectorizer = CountVectorizer()
docs = vectorizer.fit_transform(documents)
features = vectorizer.get_feature_names()
visualizer = FreqDistVisualizer(features=features, orient='v')
visualizer.fit(docs)
visualizer.show()

# plot  the EDF of the documents' length
tokenizer = RegexpTokenizer(r'\w+')
documents = [tokenizer.tokenize(doc) for doc in documents]
number_of_words = [len(doc) for doc in documents]
sns.displot(number_of_words, kind='ecdf')
plt.title("Empirical Distribution Function of Documents' Length")
plt.xlabel('Number of Words')
plt.ylabel('Percentage of Documents')
plt.grid(b=None)
plt.xlim(0, 100000)
plt.yticks([0.2, 0.4, 0.6, 0.8, 1])

# plot the number of documents on each class
unique, counts = np.unique(y_train, return_counts=True)
plt.bar(unique, counts, color='b')
unique, counts = np.unique(y_test, return_counts=True)
plt.bar(unique, counts, color='r')
plt.xticks(unique, labels, rotation='vertical')
plt.yticks([50, 100, 150, 200])
plt.title('Class Frequency')
plt.xlabel('Class')
plt.ylabel('Number of Documents')
plt.legend(['Training Documents', 'Test Documents'], loc=2)
plt.grid(b=None)
plt.show()
