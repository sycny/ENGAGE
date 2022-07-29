import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import accuracy_score, f1_score
from sklearn import preprocessing

def svc_classify(x,y):
    kf = StratifiedKFold(n_splits = 3, shuffle = True, random_state = None)
    #accuracies = []
    F1Mi = []
    F1Ma = []
    accuracies_val = []
    for train_index, test_index in kf.split(x,y):
        
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        params = {'C':[0.0001,0.001,0.01,0.1,1,10,100,1000]}
        classifier = GridSearchCV(SVC(),params, cv = 3, scoring = 'accuracy', verbose=0)
        classifier.fit(x_train,y_train)
        #accuracies.append(accuracy_score(y_test,classifier.predict(x_test)))
        F1Mi.append(f1_score(y_test,classifier.predict(x_test), average = 'micro'))
        F1Ma.append(f1_score(y_test,classifier.predict(x_test), average = 'macro'))
        
    
    return np.mean(F1Mi),np.mean(F1Ma)

'''
def eval_svm(embeddings, labels):
    
    labels = preprocessing.LabelEncoder().fit_transform(labels)
    x = embeddings.detach().cpu().numpy()
    y = labels
    
    F1mi, F1ma = svc_classify(x,y)
    
    return {'F1Mi':F1mi, 'F1Ma':F1ma}
'''
