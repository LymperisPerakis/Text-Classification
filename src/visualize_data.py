from src.train_classification_model import TrainClassificationModel
from src.configs.configs import conventional_models
import time
from typing import List
import pandas as pd
from random import randint
import seaborn as sns
from yellowbrick.text import FreqDistVisualizer
from sklearn.feature_extraction.text import CountVectorizer

model = 'fasttext'
train_model = TrainClassificationModel(model=model, process=False)
documents_list, label_list, labels, label_dummy, documents = train_model.preprocess_files()
vectorizer = CountVectorizer()
docs = vectorizer.fit_transform(documents)
features = vectorizer.get_feature_names()
visualizer = FreqDistVisualizer(features=features, orient='v')
visualizer.fit(docs)
visualizer.show()
