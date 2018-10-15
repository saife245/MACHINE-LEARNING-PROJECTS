# -*- coding: utf-8 -*-
"""
Created on Thu Oct  4 23:43:04 2018

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
features = train[['Pclass', 'Age', 'Sex', "SibSp", 'Parch','Fare', 'Embarked']].values

import keras
from keras.models import Sequential
from keras.layers import Dense

classifier = Sequential()

classifier.add(Dense(output_dim = 4, init = 'uniform', activation = 'relu', input_dim = 6))

classifier.add(Dense(output_dim = 4, init = 'uniform', activation = 'relu')) 

classifier.add(Dense(output_dim = 4, init = 'uniform', activation = 'relu')) 

classifier.add(Dense(output_dim = 4, init = 'uniform', activation = 'relu')) 

classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))

classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

classifier.fit(features, target, batch_size = 10, nb_epoch = 100)

test_features = test[["Pclass", "Age", "Sex", "SibSp", "Parch", "Fare", "Embarked"]].values
Y_pred = classifier.predict(test_features)

Y_pred = Y_pred>0.5

write_prediction(Y_pred, "saif_sub2.csv")
