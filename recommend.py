# -*- coding: utf-8 -*-

import csv
import datetime
import random
from operator import sub
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn import preprocessing, ensemble

tx = open('train_x','r')
ty = open('train_y','r')
tex = open('test_x','r')
train_X = pickle.load(tx)
train_y = pickle.load(ty)
test_X = pickle.load(tex)
tx.close()
ty.close()
tex.close()
data_path = "../../dataset/"

def runXGB(train_X, train_y, seed_val=25):
    param = {}
    param['objective'] = 'multi:softprob'
    param['eta'] = 0.08
    param['max_depth'] = 7
    param['silent'] = 1
    param['num_class'] = 22
    param['eval_metric'] = "mlogloss"
    param['min_child_weight'] = 1
    param['subsample'] = 0.9
    param['colsample_bytree'] = 0.9
    param['seed'] = seed_val
    param['gamma'] = 0.15 
    param['reg-alpha']=0.075
    num_rounds = 100

    plst = list(param.items())
    xgtrain = xgb.DMatrix(train_X, label=train_y)
    model = xgb.train(plst, xgtrain, num_rounds)	
    return model

	
print("Building model..")
model = runXGB(train_X, train_y, seed_val=0)
print("Predicting..")
xgtest = xgb.DMatrix(test_X)
preds = model.predict(xgtest)


print("Getting the top products..")
test_id = np.array(pd.read_csv(data_path + "test_ver2.csv", usecols=['ncodpers'])['ncodpers'])
new_products = []
for i, idx in enumerate(test_id):
    new_products.append([max(x1 - x2,0) for (x1, x2) in zip(preds[i,:], cust_dict[idx])])
target_cols = np.array(target_cols)
preds = np.argsort(np.array(new_products), axis=1)


#DROP product 7,8,9,12,18 as theese items have not been purchased /not popular in recent months. 
l1 = []
for i,item in enumerate(preds):
    l1.append(filter(lambda x: x not in (7,8,9,12,18),preds[i,:]))
new_pred = np.asarray(l1)
new_pred = np.fliplr(new_pred)[:,:7]
final_preds = [" ".join(list(target_cols[pred])) for pred in new_pred]
out_df = pd.DataFrame({'ncodpers':test_id, 'added_products':final_preds})
out_df.to_csv('submission.csv', index=False)
