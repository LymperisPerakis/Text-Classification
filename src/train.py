from src.make_training_files import MakeTrainingFiles
from src.classification import Classifier
from sklearn.metrics import plot_confusion_matrix
from src.configs import interface_pdf_path2, interface_text_path2
import pickle
import matplotlib.pyplot as plt

training_files = MakeTrainingFiles(pdf_path=interface_pdf_path2,
                                   text_path=interface_text_path2)
documents_list, label_list, labels, label_dummy, documents = training_files.run(process=False)

with open('training_data/train.pickle', 'wb') as f:
    pickle.dump([documents_list, label_list, labels, label_dummy, documents], f)

with open('training_data/train.pickle', 'rb') as f:
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

classifier5 = Classifier(docs=documents, label_dummy=label_dummy)  # doesnt work
text_clf5 = classifier5.train_model_LinearRegression(X_train=X_train, y_train=y_train)
accuracy5, report5, confusion_matrix5 = classifier5.test_model(text_clf5, X_test, y_test, labels)

classifier6 = Classifier(docs=documents, label_dummy=label_dummy)  # doesnt work
text_clf6 = classifier6.train_model_LDA(X_train=X_train, y_train=y_train)
accuracy6, report6, confusion_matrix6 = classifier6.test_model(text_clf6, X_test, y_test, labels)

classifier7 = Classifier(docs=documents, label_dummy=label_dummy)
text_clf7 = classifier7.train_model_LogR(X_train=X_train, y_train=y_train)
accuracy7, report7, confusion_matrix7 = classifier7.test_model(text_clf7, X_test, y_test, labels)

classifier8 = Classifier(docs=documents, label_dummy=label_dummy)
text_clf8 = classifier8.train_model_AdaB(X_train=X_train, y_train=y_train)
accuracy8, report8, confusion_matrix8 = classifier8.test_model(text_clf8, X_test, y_test, labels)

classifier9 = Classifier(docs=documents, label_dummy=label_dummy)
text_clf9 = classifier9.train_model_Rocchio(X_train=X_train, y_train=y_train)
accuracy9, report9, confusion_matrix9 = classifier9.test_model(text_clf9, X_test, y_test, labels)

classifier10 = Classifier(docs=documents, label_dummy=label_dummy)
text_clf10 = classifier10.train_model_DecisionTree(X_train=X_train, y_train=y_train)
accuracy10, report10, confusion_matrix10 = classifier10.test_model(text_clf10, X_test, y_test, labels)

X_tree = []
y_tree = []

for i, y in enumerate(y_train):
    if y == 0 or y == 1 or y == 2 or y == 3 or y == 4:
        X_tree.append(X_train[i])
        y_tree.append(y_train[i])

classifier_tree = Classifier(docs=documents, label_dummy=label_dummy)
classifier_tree.export_decision_tree_graph(X_tree, y_tree, 'tree_5_classes.dot')


fig, ax = plt.subplots(nrows=1, ncols=2)

# ## Plot roc
# for i in range(len(label_list)):
#     fpr, tpr, thresholds = metrics.roc_curve(y_test_array[:,i],
#                            predicted_prob[:,i])
#     ax[0].plot(fpr, tpr, lw=3,
#               label='{0} (area={1:0.2f})'.format(classes[i],
#                               metrics.auc(fpr, tpr))
#                )
# ax[0].plot([0,1], [0,1], color='navy', lw=3, linestyle='--')
# ax[0].set(xlim=[-0.05,1.0], ylim=[0.0,1.05],
#           xlabel='False Positive Rate',
#           ylabel="True Positive Rate (Recall)",
#           title="Receiver operating characteristic")
# ax[0].legend(loc="lower right")
# ax[0].grid(True)
#
# ## Plot precision-recall curve
# for i in range(len(classes)):
#     precision, recall, thresholds = metrics.precision_recall_curve(
#                  y_test_array[:,i], predicted_prob[:,i])
#     ax[1].plot(recall, precision, lw=3,
#                label='{0} (area={1:0.2f})'.format(classes[i],
#                                   metrics.auc(recall, precision))
#               )
# ax[1].set(xlim=[0.0,1.05], ylim=[0.0,1.05], xlabel='Recall',
#           ylabel="Precision", title="Precision-Recall curve")
# ax[1].legend(loc="best")
# ax[1].grid(True)
# plt.show()

# TODO:  confuson matrix with the down labels vertical
