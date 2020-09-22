from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import MultinomialNB

ml_models_mapping = {'SVM': SVC(), 'DecisionTree': DecisionTreeClassifier(),
                     'NaiveBayes': MultinomialNB(),
                     'RandomForest': RandomForestClassifier(),
                     'LogisticRegression': LogisticRegression(),
                     'kNN': KNeighborsClassifier(),
                     'SGD': SGDClassifier(),
                     'AdaBoost': AdaBoostClassifier(),
                     'GradientBoosting': GradientBoostingClassifier(),
                     'ExtraTrees': ExtraTreesClassifier(),
                     'Rocchio': NearestCentroid(),
                     }
