# -*- coding: utf-8 -*-
"""
Created on Sun Aug 12 02:20:12 2018

@author: MD SAIF UDDIN
Here we apply the logistic regression i.e. both linear and polynomial logistic regression  and 
analyse their differences and predict about the survival

"""
import pandas as pd
#import numpy as np
import utils

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

train['hypo'] = 0
train.loc[train.Sex == 'female', 'hypo'] = 1

train['Result'] = 0
train.loc[train.Survived == train['hypo'], 'Result'] = 1

#print(train['Result'].value_counts(normalize = True))
from sklearn import linear_model, preprocessing, model_selection
utils.clean_data(train)
utils.clean_data(test)

print(train.shape)

target = train['Survived'].values
features = train[['Pclass', 'Age','Fare', 'Embarked', 'Sex', 'SibSp', 'Parch']].values

#using logistic regression
classifier = linear_model.LogisticRegression(C = 10)
classifier = classifier.fit(features, target)
print(classifier.score(features, target))

scores = model_selection.cross_val_score(classifier, features, target, scoring = 'accuracy', cv = 50)
lin_predict = classifier.predict(scores)
utils.write_prediction(lin_predict, 'resultlogistic_regression.csv')

#here we use polynomial regression which fit much better than linear regression
poly = preprocessing.PolynomialFeatures(degree = 2)
poly_features = poly.fit_transform(features)

classifier_ = classifier.fit(poly_features, target)
print(classifier_.score(poly_features, target))

scores = model_selection.cross_val_score(classifier, features, target, scoring = 'accuracy', cv = 10)
print(scores)
print(scores.mean())

scores = poly.fit_transform(scores)
utils.write_prediction(classifier.predict(scores), 'resultlogistic_regression_poly.csv')
