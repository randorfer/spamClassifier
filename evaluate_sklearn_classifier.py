'''
train based on a decision tree
'''
import operator

from sklearn.tree import DecisionTreeClassifier, export_graphviz
import numpy as np
from sklearn.externals.six import StringIO
from sklearn.metrics import roc_curve, auc, confusion_matrix, precision_recall_fscore_support, classification_report
from sklearn.model_selection import RandomizedSearchCV
import pydot_ng
import matplotlib.pyplot as plt

from load_data import load_data

__labels__ = [
    "make", "address", "all", "3d", "our", "over", "remove", "internet",
    "order", "mail", "receive", "will", "people", "report", "addresses",
    "free", "business", "email", "you", "credit", "your", "font", "000",
    "money", "hp", "hpl", "george", "650", "lab", "labs", "telnet", "857",
    "data", "415", "85", "technology", "1999", "parts", "pm", "direct", "cs",
    "meeting", "original", "project", "re", "edu", "table", "conference", ";",
    "(", "[", "!", "$", "#", "avg_capital_run_length", "longest_capital_run_length",
    "total_capital_run_length"
]

def top_n_features(number, weights):
    '''
    docstring
    '''
    return sorted(zip(__labels__, weights), reverse=True, key=operator.itemgetter(1))[:number]
def evaluate_classifier(clf, number_of_top_features=5, fscore_beta=1.0):
    '''
    docstring
    '''
    X_train, X_test, y_train, y_test = load_data(train=True, test_size=0.4)
#    clf=tune(clf,X_train, y_train, tuned_parameters)
    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)
    y_pred = clf.predict(X_test)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test,y_pred)
    precision, recall, fbeta_score, support = precision_recall_fscore_support(
        y_true=y_test, y_pred=y_pred,beta=fscore_beta, labels=1, average='binary'
    )
    if hasattr(clf, "predict_proba"):
        false_positive_rate, true_positive_rate, _ = roc_curve(
            y_test, clf.predict_proba(X_test)[:, 1]
        )
        roc_auc = auc(false_positive_rate, true_positive_rate)
    else:
        false_positive_rate, true_positive_rate, _ = roc_curve(
            y_test, clf.decision_function(X_test)
        )
        roc_auc = auc(false_positive_rate, true_positive_rate)

    if hasattr(clf, "feature_importances_"):
        top_features = top_n_features(number=number_of_top_features, weights=clf.feature_importances_)
    else:
        top_features = (-1,-1)

    return score, roc_auc, top_features, conf_matrix, precision, recall, fbeta_score, support, class_report