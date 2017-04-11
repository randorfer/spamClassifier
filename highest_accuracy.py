'''
'''

from evaluate_sklearn_classifier import evaluate_classifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
__noteable_features__ = []
def main(beta=1.0):
    '''
    '''
    classifiers = {
        "Multi-NB": MultinomialNB(alpha=1.0),
        "DecisionTree": DecisionTreeClassifier(criterion="gini"),
        "RandomForest": RandomForestClassifier(criterion="gini", max_features="log2", n_estimators=73),
        "KNeighbors": KNeighborsClassifier(n_neighbors=5),
        "SVC": SVC(),
        "AdaBoost": AdaBoostClassifier(n_estimators=50, learning_rate=1.0, algorithm='SAMME.R'),
    }
    results = {}
    for classifier_name, clf in classifiers.iteritems():
        print "evaluating %s" % classifier_name
        classification_result = evaluate_classifier(clf, classifier_name, number_of_top_features=10,fscore_beta=beta)
        results[classifier_name] = classification_result
        print 'Accuracy: %0.5f, AUC: %0.5f' % (classification_result[0], classification_result[1])
        print classification_result[3]
        print 'FScore: %0.5f, Precision: %0.5f, Recall: %0.5f' % (
            classification_result[6],
            classification_result[4],
            classification_result[5]
        )
        print 'classification report'
        print classification_result[8]
        index = 1
        for feature in classification_result[2]:
            if feature != -1:
                if feature[0] not in __noteable_features__:
                    __noteable_features__.append(feature[0])
                print "Top [%s] feature: Name [%s] weight [%0.5f]" % (index, feature[0], feature[1])
                index+=1

if __name__ == "__main__":
    '''
    '''
    main()
    main(10)
    print __noteable_features__
