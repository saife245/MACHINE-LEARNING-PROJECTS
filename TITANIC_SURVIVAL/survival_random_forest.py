# -*- coding: utf-8 -*-
"""
Created on Mon Aug 13 11:02:17 2018

@author: MD SAIF UDDIN

here we apply random forrest on dataset to analyse the difference with
the pridiction on random forest and decision tree
"""
import pandas as pd
#import numpy as np
import utils

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

utils.clean_data(train)
utils.clean_data(test)

print(train.shape)
target = train['Survived'].values
features_forest = train[['Pclass', 'Age', 'Sex', 'SibSp', 'Parch', 'Embarked']].values

#USE random forest classifier
from sklearn import ensemble, model_selection
forest = ensemble.RandomForestClassifier(
        max_depth = 7,
        min_samples_split = 4,
        n_estimators = 1000,
        n_jobs = -1,
       random_state = 1
        )
forest = forest.fit(features_forest, target)

print(forest.feature_importances_)
print(forest.score(features_forest, target))

scores = model_selection.cross_val_score(forest, features_forest, target, scoring = 'accuracy', cv = 10)
print(scores)
print(scores.mean())

test_features_forest = test[["Pclass", "Age", "Sex", "SibSp", "Parch", "Embarked"]].values
prediction_forest = forest.predict(test_features_forest)
utils.write_prediction(prediction_forest, "resultsrandom_forest.csv")
