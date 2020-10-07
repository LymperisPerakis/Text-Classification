from gensim.parsing.preprocessing import preprocess_string
from gensim import corpora, models
from src.classification.train_classification_model import \
    TrainClassificationModel
import pickle
import pyLDAvis
from pyLDAvis import gensim

# load preprocessed files
model = 'fasttext'
file = 'train/cleaned_21_on_vocab.pickle'
train_model = TrainClassificationModel(model=model)
documents_list, label_list, labels, label_dummy, documents = \
    train_model.load_files(file)

# create dictionary for topic modeling
processing = []
for i, document in enumerate(documents):
    processing.append(preprocess_string(document))
dictionary = corpora.Dictionary(processing)

corpus = [dictionary.doc2bow(text) for text in processing]

# transform with TFIDF
tfidf = models.TfidfModel(corpus)

transformed_tfidf = tfidf[corpus]
num_topics = 20
lda = models.LdaMulticore(transformed_tfidf, num_topics=num_topics,
                          id2word=dictionary)

# save the files
with open('train/lda.pickle', 'wb') as f:
    pickle.dump([processing, lda, corpus, dictionary], f)

# visualize and save as html
vis = gensim.prepare(lda, corpus, dictionary)
pyLDAvis.save_html(vis, f'lda{num_topics}.html')
