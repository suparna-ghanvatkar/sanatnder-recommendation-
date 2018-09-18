# -*- coding: utf-8 -*-
# Parameter tunning in Xtreme gradient Boosting

import sys
import math
 
import numpy as np
from sklearn.grid_search import GridSearchCV

import xgboost as xgb
from data_prep import train_X, train_y, test_X

tx = open('train_x','r')
ty = open('train_y','r')
tex = open('test_x','r')
train_X = pickle.load(tx)
train_y = pickle.load(ty)
test_X = pickle.load(tex)
tx.close()
ty.close()
tex.close()

class XGBoostClassifier():
    def __init__(self, num_boost_round=10, **params):
        self.clf = None
        self.num_boost_round = num_boost_round
        self.params = params
        self.params.update({'objective': 'multi:softprob'})
 
    def fit(self, X, y, num_boost_round=None):
        num_boost_round = num_boost_round or self.num_boost_round
        self.label2num = dict((label, i) for i, label in enumerate(sorted(set(y))))
        dtrain = xgb.DMatrix(X, label=[self.label2num[label] for label in y])
        self.clf = xgb.train(params=self.params, dtrain=dtrain, num_boost_round=num_boost_round)
 
    def predict(self, X):
        num2label = dict((i, label)for label, i in self.label2num.items())
        Y = self.predict_proba(X)
        y = np.argmax(Y, axis=1)
        return np.array([num2label[i] for i in y])
 
    def predict_proba(self, X):
        dtest = xgb.DMatrix(X)
        return self.clf.predict(dtest)
 
    def score(self, X, y):
        Y = self.predict_proba(X)
        return 1 / logloss(y, Y)
 
    def get_params(self, deep=True):
        return self.params
 
    def set_params(self, **params):
        if 'num_boost_round' in params:
            self.num_boost_round = params.pop('num_boost_round')
        if 'objective' in params:
            del params['objective']
        self.params.update(params)
        return self
    
    
def logloss(y_true, Y_pred):
    label2num = dict((name, i) for i, name in enumerate(sorted(set(y_true))))
    return -1 * sum(math.log(y[label2num[label]]) if y[label2num[label]] > 0 else -np.inf for y, label in zip(Y_pred, y_true)) / len(Y_pred)


def main():
    clf = XGBoostClassifier(
        eval_metric = 'logloss',
        num_class = 22,
        nthread = 4,
        eta = 0.07,
        num_boost_round = 10,
        max_depth = 9,
        min_child_weight = 1,
        subsample = 0.85,
        colsample_bytree = 0.8,
        silent = 1,
        reg_alpha=0.07,
        gamma = 0.9
        )
    parameters = {
        'gamma':[0.6, 0.7,0.8,0.9,1]
        #'num_boost_round': [50, 100, 150],
        #'eta': [0.06,0.065,0.07],
        #'reg_alpha':[0.067,0.07, 0.072,0.075]
        #'max_depth': [7,8,9],
        #'min_child_weight':[1,2,3]
        #'subsample': [0.9, 0.8, 0.85, 0.95],
        #'colsample_bytree': [0.8,0.85, 0.9, 0.95],
    }
    clf = GridSearchCV(clf, parameters, n_jobs=1, cv=2)
    print("Fitting the model")
    clf.fit(train_X,train_y)
    print("Training model completed")
    best_parameters, score, _ = max(clf.grid_scores_, key=lambda x: x[1])
    print(score)
    for param_name in sorted(best_parameters.keys()):
        print("%s: %r" % (param_name, best_parameters[param_name]))
                
    print(clf.predict(test_X))


if __name__ == '__main__':
    start_time=datetime.datetime.now()
    main()
    print(datetime.datetime.now()-start_time)
