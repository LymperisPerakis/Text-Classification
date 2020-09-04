from src.make_training_files import MakeTrainingFiles
from src.classification import Classifier
from sklearn.metrics import plot_confusion_matrix
from src.configs import interface_pdf_path, interface_text_path
import pickle
from src.transformer import MultiHeadSelfAttention, TransformerBlock, TokenAndPositionEmbedding

# training_files = MakeTrainingFiles(pdf_path=interface_pdf_path,
#                                    text_path=interface_text_path)
# documents_list, label_list, labels, label_dummy, documents = training_files.run()
#
# with open('train.pickle', 'wb') as f:
#     pickle.dump([documents_list, label_list, labels, label_dummy, documents], f)

with open('train.pickle', 'rb') as f:
    documents_list, label_list, labels, label_dummy, documents = pickle.load(f)


classifier = Classifier(docs=documents, label_dummy=label_dummy)
X_train, X_test, y_train, y_test = classifier.split_set(test_size=0.2, random_state=0)
text_clf = classifier.train_model_SVM(X_train=X_train, y_train=y_train)
accuracy, report, confusion_matrix = classifier.test_model(text_clf, X_test, y_test, labels)

classifier2 = Classifier(docs=documents, label_dummy=label_dummy)
text_clf2 = classifier2.train_model_Bayes(X_train=X_train, y_train=y_train)
accuracy2, report2, confusion_matrix2 = classifier2.test_model(text_clf2, X_test, y_test, labels)

classifier3 = Classifier(docs=documents, label_dummy=label_dummy)
text_clf3 = classifier3.train_model_KNClassifier(X_train=X_train, y_train=y_train)
accuracy3, report3, confusion_matrix3 = classifier3.test_model(text_clf3, X_test, y_test, labels)

classifier4 = Classifier(docs=documents, label_dummy=label_dummy)
text_clf4 = classifier4.train_model_RandomForest(X_train=X_train, y_train=y_train)
accuracy4, report4, confusion_matrix4 = classifier4.test_model(text_clf4, X_test, y_test, labels)

classifier5 = Classifier(docs=documents, label_dummy=label_dummy) # doesnt work
text_clf5 = classifier5.train_model_LinearRegression(X_train=X_train, y_train=y_train)
accuracy5, report5, confusion_matrix5 = classifier5.test_model(text_clf5, X_test, y_test, labels)

classifier6 = Classifier(docs=documents, label_dummy=label_dummy) # doesnt work
text_clf6 = classifier6.train_model_LDA(X_train=X_train, y_train=y_train)
accuracy6, report6, confusion_matrix6 = classifier6.test_model(text_clf6, X_test, y_test, labels)

classifier7 = Classifier(docs=documents, label_dummy=label_dummy)
text_clf7 = classifier7.train_model_LogR(X_train=X_train, y_train=y_train)
accuracy7, report7, confusion_matrix7 = classifier7.test_model(text_clf7, X_test, y_test, labels)