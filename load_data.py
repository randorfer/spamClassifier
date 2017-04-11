# -*- coding: utf-8 -*-
'''
Loads the spambase dataset.
Optionally sets to train which will return a split dataset
'''
import csv
import numpy as np
from numpy.random import RandomState
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler

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
        return train_test_split(X, y, test_size=test_size, random_state=RandomState())
    else:
        return X, y

if __name__ == '__main__':
    '''
    '''
    print load_data()
