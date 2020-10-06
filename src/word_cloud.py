import matplotlib.pyplot as plt
from wordcloud import WordCloud

from src.classification.train_classification_model import \
    TrainClassificationModel

# load documents and prepare the for the wordcloud
file = 'train/cleaned_21_on_vocab.pickle'
train_model = TrainClassificationModel(model='fasttext', process=False)
documents_list, label_list, labels, label_dummy, documents = \
    train_model.load_files(file)
comment_words = ' '.join(documents)

# define wordcloud
wordcloud = WordCloud(width=800, height=800,
                      background_color='white',
                      collocations=True,
                      min_font_size=10).generate(comment_words)

# plot the WordCloud image
plt.figure(figsize=(8, 8), facecolor=None)
plt.imshow(wordcloud)
plt.axis("off")
plt.tight_layout(pad=0)
plt.show()
