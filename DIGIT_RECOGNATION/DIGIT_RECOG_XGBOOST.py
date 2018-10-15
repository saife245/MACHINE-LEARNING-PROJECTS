# -*- coding: utf-8 -*-
"""
Created on Thu Oct 11 00:06:05 2018
__Kaggle DIGIT RECOGNATION BY XGBOOST___________
@author: MD SAIF UDDIN
"""

import pandas as pd
train = pd.read_csv("train.csv").as_matrix()
X = train[:, 1:]
Y = train[:, 0]

test = pd.read_csv("test.csv").as_matrix()

from xgboost import XGBClassifier
forest = XGBClassifier()
forest = forest.fit(X, Y)

print(forest.score(X, Y))

from sklearn.model_selection import cross_val_score
scores = cross_val_score(forest, X, Y, scoring = 'accuracy', cv = 10)
print(scores)
print(scores.mean())


prid_forest = forest.predict(test)

import numpy as np
sam = pd.read_csv("sample_submission.csv")
def write_prediction(prediction, name):
    ImageId = np.array(sam['ImageId']).astype(int)
    solution = pd.DataFrame(prediction, ImageId, columns = ['Label'])
    solution.to_csv(name, index_label = ['ImageId'])
    
write_prediction(prid_forest, "samdigit_xgboost.csv")