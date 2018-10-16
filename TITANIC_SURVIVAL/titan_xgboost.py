# -*- coding: utf-8 -*-
"""
Created on Tue Oct  9 03:19:55 2018

@author: MD SAIF UDDIN
"""

import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
#import utils

def clean_data(data):
    data['Fare'] = data['Fare'].fillna(data['Fare'].dropna().median())
    data['Age'] =  data['Age'].fillna(data['Age'].dropna().median())
    
    data.loc[data['Sex'] == 'male', 'Sex'] = 0
    data.loc[data['Sex'] =='female',  'Sex'] = 1
    
    data['Embarked'] = data['Embarked'].fillna('S')
    data.loc[data["Embarked"] == 'S', 'Embarked'] = 0
    data.loc[data['Embarked'] == 'C', 'Embarked'] = 1
    data.loc[data['Embarked'] == 'Q', 'Embarked'] =2
    

def write_prediction(prediction, name):
    PassengerId = np.array(test['PassengerId']).astype(int)
    solution = pd.DataFrame(prediction, PassengerId, columns = ['Survived'])
    solution.to_csv(name, index_label = ['PassengerId'])
    

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

#cleaning the data
clean_data(train)
clean_data(test)

target = train['Survived'].values
features = train[['Pclass', 'Age', 'Sex', "SibSp", 'Parch', 'Fare', 'Embarked']].values

from xgboost import XGBClassifier
forest = XGBClassifier()
forest = forest.fit(features, target)

print(forest.feature_importances_)
print(forest.score(features, target))

from sklearn.model_selection import cross_val_score
scores = cross_val_score(forest, features, target, scoring = 'accuracy', cv = 10)
print(scores)
print(scores.mean())

test_features_forest = test[["Pclass", "Age", "Sex", "SibSp", "Parch",'Fare', "Embarked"]].values
prediction_forest = forest.predict(test_features_forest)
write_prediction(prediction_forest, "saif_sub1.csv")