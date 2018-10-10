# -*- coding: utf-8 -*-
"""
Created on Sat Aug 11 23:37:59 2018
@author: MD SAIF UDDIN

analysis the every thing(i.e. gender, age, rich ,poor, class, children ) with the help of ploting graph
"""

import pandas as pd
import matplotlib.pyplot as plt
#import numpy as np

#importing the dataset
train = pd.read_csv("train.csv")
test = pd.read_csv('test.csv')

fig = plt.figure(figsize = (6, 4))

#normalize for percentage and kind is used to type of graph and alpha is used as color contest
#it tell what percentage of people survived and what about are dead
plt.subplot2grid((2, 3), (0, 0))
train.Survived.value_counts(normalize = True).plot(kind = "bar", alpha = 0.8)
plt.title("survived")

#it tell about the relation between the age and surviver
plt.subplot2grid((2, 3), (0, 1))
plt.scatter(train.Survived, train.Age, alpha = 0.2)
plt.title("Age wrt Survived")

#it tell the about the people belong to which class 1st class or 2nd class
plt.subplot2grid((2, 3), (0, 2))
train.Pclass.value_counts(normalize = True).plot(kind = "bar", alpha = 0.8)
plt.title("class")


plt.subplot2grid((2, 3), (1, 0), colspan = 2)
for x in [1, 2, 3]:
    train.Age[train.Pclass == x].plot(kind = 'kde')#kde stand for kernel density estimation
plt.title("class wrt Age")
plt.legend(('1st', '2nd', '3rd'))

#tell where the passenger come from
plt.subplot2grid((2, 3), (1, 2))
train.Embarked.value_counts(normalize = True).plot(kind = 'bar', alpha = 0.8)

#analysis of survival on the basis of gender
plt.subplot2grid((3,4), (0, 1))
train.Survived[train.Sex == "male"].value_counts(normalize = True).plot(kind = 'bar', alpha = 0.8)
plt.title('Men survived')

female_color = '#FA0000'
plt.subplot2grid((3, 4), (0, 2))
train.Survived[train.Sex == 'female'].value_counts(normalize = True).plot(kind = 'bar', alpha = 0.8, color = female_color)
plt.title("Female survived")

plt.subplot2grid((3, 4), (0, 3))
train.Sex[train.Survived == 1].value_counts(normalize = True).plot(kind = 'bar', alpha = 0.8, color = [female_color, 'b'])
plt.title("Sex Survived")

plt.subplot2grid((3, 4), (1, 0), colspan = 4)
for x in [1, 2, 3]:
    train.Survived[train.Pclass == x].plot(kind = 'kde')#kde stand for kernel density estimation
plt.title("class wrt Survived")
plt.legend(('1st', '2nd', '3rd'))

plt.subplot2grid((3,4), (2, 0))
train.Survived[(train.Sex == "male") & (train.Pclass == 1)].value_counts(normalize = True).plot(kind = 'bar', alpha = 0.8)
plt.title('Rich Men survived')

plt.subplot2grid((3,4), (2, 1))
train.Survived[(train.Sex == "male") & (train.Pclass == 3)].value_counts(normalize = True).plot(kind = 'bar', alpha = 0.8)
plt.title('poor Men survived')

plt.subplot2grid((3,4), (2, 2))
train.Survived[(train.Sex == "female") & (train.Pclass == 1)].value_counts(normalize = True).plot(kind = 'bar', alpha = 0.8)
plt.title('Rich WoMen survived')

plt.subplot2grid((3,4), (2, 3))
train.Survived[(train.Sex == "female") & (train.Pclass == 3)].value_counts(normalize = True).plot(kind = 'bar', alpha = 0.8)
plt.title('Poor WoMen survived')

plt.show()

#About children in this tragedy
train["Child"] = float('NaN')
train.loc[train["Age"] >= 18, "Child"] = 0
train.loc[train["Age"] < 18, "Child"] = 1

print ("\nChildren survived / children passed")
print(train["Survived"][train["Child"] == 1].value_counts(normalize = True))

print ("\nPredict 1 if female and 0 if male")
test_one = test
test_one["Survived"] = 0
test_one.loc[test_one["Sex"] == 'female', "Survived"] = 1
test_one.to_csv("results/gender_model.csv", index = False, columns = ["PassengerId", "Survived"])

print ("\nCheck how accurate this model is on training set")
train["Hyp"] = 0
train.loc[train["Sex"] == "female", "Hyp"] = 1

train["Result"] = 0
train.loc[train["Hyp"] == train["Survived"], "Result"] = 1
print(train["Result"].value_counts(normalize = True))
