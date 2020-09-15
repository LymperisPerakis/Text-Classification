from src.classification import Classifier
from src.make_training_files import MakeTrainingFiles
from src.configs import interface_pdf_path, interface_text_path
import fasttext
import numpy as np

training_files = MakeTrainingFiles(pdf_path=interface_pdf_path,
                                   text_path=interface_text_path)

documents_list, label_list, labels, label_dummy, documents = \
    training_files.run('lem', True)

training_data = []

for i in range(len(documents)):
    training_data.append('__label__'+str(label_dummy[i])+ ' ' + documents[i])

classifier = Classifier(docs=training_data, label_dummy=label_dummy)
X_train, X_test, y_train, y_test = classifier.split_set(test_size=0.2,
                                                        random_state=0)

classifier = Classifier(docs=documents, label_dummy=label_dummy)
_, X_test2, _, _ = classifier.split_set(test_size=0.2, random_state=0)

with open('training_data/fasttext_train.txt', 'w') as filehandle:
    filehandle.writelines("%s\n" % document for document in X_train)

with open('training_data/fasttext_test.txt', 'w') as filehandle:
    filehandle.writelines("%s\n" % document for document in X_test)



model = fasttext.train_supervised('training_data/fasttext_train.txt', epoch=100, lr=1, wordNgrams=3)

def print_results(N, p, r):
    print("N\t" + str(N))
    print("P@{}\t{:.3f}".format(1, p))
    print("R@{}\t{:.3f}".format(1, r))

print_results(*model.test('training_data/fasttext_test.txt'))

predicted = model.predict(X_test2)
pred = [int(p[0].split('__label__')[1]) for p in predicted[0]]
accuracy = np.mean(np.array(pred) == y_test)
print(f'The accuracy of the model is: {accuracy*100}% ')
