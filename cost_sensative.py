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
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import csv
import numpy as np
from numpy.random import RandomState
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
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
def load_data(
        train=False,
        test_size=0.4
    ):
    data = []

    # Read the training data
    file_handle = open('data/spambase.data')
    reader = csv.reader(file_handle)
    next(reader, None)
    for row in reader:
        data.append(row)
    file_handle.close()

    X = np.array([x[:-1] for x in data]).astype(np.float)
    scaler = StandardScaler()
    scaler.fit(X)

    # The final column is the target (spam == 1, ham ==0)
    y = np.array([x[-1] for x in data]).astype(np.float)
    del data # free up the memory

    if train:
        # returns X_train, X_test, y_train, y_test
        X_train, X_test, y_train, y_test, cost_mat_train, cost_mat_test = \
        train_test_split(X, data.target, data.cost_mat, test_size=0.33, random_state=10)
        return train_test_split(X, y, cost_mat_test=() test_size=test_size, random_state=RandomState())
    else:
        return X, y

def main():
    X_train, X_test, y_train, y_test = load_data(train=True, test_size=0.4)
    classifiers = {"RF": {"f": RandomForestClassifier()},
                   "DT": {"f": DecisionTreeClassifier()}}
    ci_models = ['DT', 'RF']
    # Fit the classifiers using the training dataset
    for model in classifiers.keys():
        classifiers[model]["f"].fit(X_train, y_train)
        classifiers[model]["c"] = classifiers[model]["f"].predict(X_test)
        classifiers[model]["p"] = classifiers[model]["f"].predict_proba(X_test)
        classifiers[model]["p_train"] = classifiers[model]["f"].predict_proba(X_train)
    from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

    measures = {"F1Score": f1_score, "Precision": precision_score, 
                "Recall": recall_score, "Accuracy": accuracy_score}
    results = pd.DataFrame(columns=__labels__)

    
    from costcla.models import BayesMinimumRiskClassifier

    for model in ci_models:
        classifiers[model+"-BMR"] = {"f": BayesMinimumRiskClassifier()}
        # Fit
        classifiers[model+"-BMR"]["f"].fit(y_test, classifiers[model]["p"])
        # Calibration must be made in a validation set
        # Predict
        classifiers[model+"-BMR"]["c"] = classifiers[model+"-BMR"]["f"].predict(classifiers[model]["p"], cost_mat_test)
if __name__ == '__main__':
    main()