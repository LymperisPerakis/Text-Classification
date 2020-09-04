from src.classification import Classifier
import pickle
import ktrain
from ktrain import text
from src.make_training_files import MakeTrainingFiles
from src.configs import interface_pdf_path, interface_text_path

training_files = MakeTrainingFiles(pdf_path=interface_pdf_path,
                                   text_path=interface_text_path)
documents_list, label_list, labels, label_dummy, documents = \
    training_files.get_unprocessed()
classifier = Classifier(docs=documents, label_dummy=label_dummy)
X_train, X_test, y_train, y_test = classifier.split_set(test_size=0.2,
                                                        random_state=0)

(x_train, y_train), (x_test, y_test), preproc = text.texts_from_array(
    x_train=X_train, y_train=y_train,
    x_test=X_test, y_test=y_test,
    class_names=label_list,
    preprocess_mode='bert',
    maxlen=512,
    max_features=35000)

with open('train_bert.pickle', 'wb') as f:
    pickle.dump([x_train, y_train, x_test, y_test, preproc, label_list], f)

with open('train_bert.pickle', 'rb') as f:
    x_train, y_train, x_test, y_test, preproc, label_list = pickle.load(f)

model = text.text_classifier('bert', train_data=(x_train, y_train),
                             preproc=preproc)
learner = ktrain.get_learner(model, train_data=(x_train, y_train),
                             batch_size=6)

learner.fit_onecycle(2e-5, 4)
learner.validate(val_data=(x_test, y_test), class_names=label_list)
