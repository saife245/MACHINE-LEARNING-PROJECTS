# -*- coding: utf-8 -*-
"""
Created on Sun Aug 12 13:25:44 2018

@author: MD SAIF UDDIN

analysis the dataset with the help of decision tree

"""

import pandas as pd
import utils
from sklearn import tree, model_selection

train = pd.read_csv("train.csv")
test =pd.read_csv('test.csv')

utils.clean_data(train)
utils.clean_data(test)

target = train['Survived'].values
features = train[['Pclass', 'Age','Fare', 'Embarked', 'Sex', 'SibSp', 'Parch']].values

decision_tree = tree.DecisionTreeClassifier(random_state = 1)
decision_tree = decision_tree.fit(features, target)

print(decision_tree.feature_importances_)
print(decision_tree.score(features, target))

#let's trry on test set
test_features = test[['Pclass', 'Age','Fare', 'Embarked', 'Sex', 'SibSp', 'Parch']].values
prediction = decision_tree.predict(test_features)
utils.write_prediction(prediction, "decision123_tree.csv")

'''above case is the case of overfitting
 now let correct this over fitting
 '''
scores = model_selection.cross_val_score(decision_tree, features, target, scoring = 'accuracy', cv = 50)
print(scores)
print(scores.mean())

generalized_tree = tree.DecisionTreeClassifier(
        max_depth = 7,
        min_samples_split = 2,
        random_state = 1
        )
generalized_tree = generalized_tree.fit(features, target)
print(generalized_tree.feature_importances_)
print(generalized_tree.score(features, target))

scores = model_selection.cross_val_score(generalized_tree, features, target, scoring = 'accuracy', cv = 50)
print(scores)
print(scores.mean())

#writing new pridiction
test_features_two = test[['Pclass', 'Age','Fare', 'Embarked', 'Sex', 'SibSp', 'Parch']].values
prediction_two = generalized_tree.predict(test_features_two)
utils.write_prediction(prediction_two, "resultsdecision_tree_two.csv")

#creating tree.dot file which visualize the raandom forest and who it work
tree.export_graphviz(generalized_tree, out_file = 'tree.dot')
