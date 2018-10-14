
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 11 00:06:05 2018
___KAGGLE DIGIT RECOGNATION BY RANDOM FOREST___________
@author: MD SAIF UDDIN
"""

import pandas as pd
train = pd.read_csv("train.csv").as_matrix()
X = train[:, 1:]
Y = train[:, 0]

test = pd.read_csv("test.csv")

from sklearn import ensemble, model_selection
forest = ensemble.RandomForestClassifier(
        n_estimators = 100,
        max_depth = 8,
        min_samples_split = 4,
        n_jobs = -1,
        random_state = 1 )
forest = forest.fit(X, Y)

print(forest.score(X, Y))

scores = model_selection.cross_val_score(forest, X, Y, scoring = 'accuracy', cv = 10, n_jobs = -1)
print(scores)
print(scores.mean())


prid_forest = forest.predict(test)

import numpy as np
sam = pd.read_csv("sample_submission.csv")
def write_prediction(prediction, name):
    ImageId = np.array(sam['ImageId']).astype(int)
    solution = pd.DataFrame(prediction, ImageId, columns = ['Label'])
    solution.to_csv(name, index_label = ['ImageId'])
    
write_prediction(prid_forest, "sam_digit.csv")
