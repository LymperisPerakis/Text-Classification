from src.classification import Classifier
import pickle
from src.transformer import MultiHeadSelfAttention, TransformerBlock, TokenAndPositionEmbedding
import keras

with open('train.pickle', 'rb') as f:
    documents_list, label_list, labels, label_dummy, documents = pickle.load(f)
classifier = Classifier(docs=documents, label_dummy=label_dummy)
X_train, X_test, y_train, y_test = classifier.split_set(test_size=0.2, random_state=0)


maxlen = 10000  # Only consider the first 200 words of each movie review
vocab_size = 200000  # Only consider the top 20k words

x_train = keras.preprocessing.sequence.pad_sequences(X_train, maxlen=maxlen)