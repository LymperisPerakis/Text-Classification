from src.classification import Classifier
import pickle
import ktrain
from ktrain import text
from src.make_training_files import MakeTrainingFiles
from src.configs import interface_pdf_path, interface_text_path

training_files = MakeTrainingFiles(pdf_path=interface_pdf_path,
                                   text_path='Text Data/Interface ICs Text')
documents_list, label_list, labels, label_dummy, documents = \
    training_files.run(process=False)
classifier = Classifier(docs=documents, label_dummy=label_dummy)
X_train, X_test, y_train, y_test = classifier.split_set(test_size=0.2,
                                                        random_state=0)
# TODO: test with Roberta
MODEL_NAME = 'roberta-base'
t = text.Transformer(MODEL_NAME, maxlen=500, class_names=labels)
trn = t.preprocess_train(X_train, y_train)
val = t.preprocess_test(X_test, y_test)
model_roberta = t.get_classifier()
learner_roberta = ktrain.get_learner(model_roberta, train_data=trn, val_data=val, batch_size=2)
learner_roberta.lr_find(show_plot=True, max_epochs=2)
learner_roberta.fit_onecycle(2e-5, 4)
learner_roberta.validate(val_data=val, class_names=labels)

with open('training_data/train.pickle', 'rb') as f:
    documents_list, label_list, labels, label_dummy, documents = pickle.load(f)


(x_train, y_train), (x_test, y_test), preproc = text.texts_from_array(
    x_train=X_train, y_train=y_train,
    x_test=X_test, y_test=y_test,
    class_names=label_list,
    preprocess_mode='bert',
    maxlen=512,
    max_features=35000)

with open('training_data/train_bert.pickle', 'wb') as f:
    pickle.dump([x_train, y_train, x_test, y_test, preproc, label_list], f)

with open('training_data/train_bert.pickle', 'rb') as f:
    x_train, y_train, x_test, y_test, preproc, label_list = pickle.load(f)

model = text.text_classifier('bert', train_data=(x_train, y_train),
                             preproc=preproc)
learner = ktrain.get_learner(model, train_data=(x_train, y_train),
                             batch_size=2)

learner.fit_onecycle(2e-5, 4)
learner.validate(val_data=(x_test, y_test), class_names=labels)

