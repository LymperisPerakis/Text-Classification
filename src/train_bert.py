from src.classification.classification import Classifier
import pickle
import ktrain
from ktrain import text
from transformers import TFAutoModelForSequenceClassification
import fasttext
import numpy as np

#training_files = MakeTrainingFiles(pdf_path=interface_pdf_path,
#                                   text_path='Text Data/Interface ICs Text')
#documents_list, label_list, labels, label_dummy, documents = \
#    training_files.run(process=False)

with open('training_data/cleaned_docs_25_classes_on_vocab_lem.pickle', 'rb') as f:
    documents_list, label_list, labels, label_dummy, documents = pickle.load(f)

classifier = Classifier(docs=documents, label_dummy=label_dummy)
X_train, X_test, y_train, y_test = classifier.split_set(test_size=0.2,
                                                        random_state=0)
# TODO: test with Roberta
MODEL_NAME = 'roberta-base'
#MODEL_NAME = 'gpt2'
MODEL_NAME = 'xlnet-base-cased'
t = text.Transformer(MODEL_NAME, maxlen=500, class_names=labels)
trn = t.preprocess_train(X_train, y_train)
val = t.preprocess_test(X_test, y_test)
with open('training_data/trn_and_val_xlnet.pickle', 'wb') as f:
    pickle.dump([trn, val], f)
with open('training_data/trn_and_val_xlnet.pickle', 'rb') as f:
    trn, val = pickle.load(f)

model_roberta = t.get_classifier()
learner_roberta = ktrain.get_learner(model_roberta, train_data=trn, val_data=val, batch_size=5) # for xlnet large batch_size 1
#learner_roberta.lr_find(show_plot=True, max_epochs=2)
learner_roberta.fit_onecycle(2e-5, 6)

# save model
learner_roberta.model.save_pretrained('training_data/learner_xlnet')
learner = TFAutoModelForSequenceClassification.from_pretrained('training_data/learner_xlnet')

# validate
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

