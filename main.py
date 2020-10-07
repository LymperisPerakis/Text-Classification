from src.classification.train_classification_model import \
    TrainClassificationModel
from src.configs.configs import conventional_models, transformers_models, \
    text_path_interface

# choose model between conventional, transformers and fasttext
model = conventional_models[0]
# model = transformers_models[0]
# model = 'fasttext'

# whether to load the files or process them
load = True
file_path = 'train/cleaned_21.pickle'

# initialize Classsification class
train_model = TrainClassificationModel(model=model,
                                       text_path=text_path_interface,
                                       process=True,
                                       lem_or_stem='lem',
                                       on_vocab=False,
                                       preprocessing_tool='NLTK')

# load or process the documents
if load:
    documents_list, label_list, labels, label_dummy, documents = \
        train_model.load_files(file_path)
else:
    documents_list, label_list, labels, label_dummy, documents = \
        train_model.preprocess_files()

# split the documents in training and test set
if model != 'fasttext':
    X_train, X_test, y_train, y_test = \
        train_model.split_training_files(test_size=0.2, random_state=0)

# train model
if model == 'fasttext':
    kwargs = {'test_size': 0.2, 'random_state': 0,
              'epoch': 80, 'lr': 1, 'wordNgrams': 3}
    fitted_model = train_model.train_model(**kwargs)
else:
    fitted_model = train_model.train_model()

# validate model
accuracy, report, confusion_matrix, predicted, precision, recall, f1, \
balanced_accuracy = train_model.validate_model()
