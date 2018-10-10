# -*- coding: utf-8 -*-
"""
Created on Mon Aug 13 15:40:46 2018

@author: MD SAIF UDDIN
we applying the logistic regression on given training set and analyse about survival,
By ploting .
"""

import pandas as pd
import statsmodels.api as sm
from patsy import dmatrices
import matplotlib.pyplot as plt
#from sklearn import model_selection

# load data
train = pd.read_csv("train.csv")

# clear data
train = train.drop(['Ticket','Cabin'], axis=1)
train.Age = train.Age.interpolate()
train.Embarked = train.Embarked.fillna('S')

# run logistic regression
formula = "Survived ~ C(Pclass) + C(Sex) + Age + SibSp + C(Embarked)"
results = {}

y,x = dmatrices(formula, data=train, return_type="dataframe")

model = sm.Logit(y,x)
res = model.fit()

results["Logit"] = [res, formula]
print(res.summary())

# print some stats
plt.figure(figsize=(18,4))

plt.subplot(121)
ypred = res.predict(x)
plt.plot(x.index, ypred, 'bo', x.index, y, 'mo', alpha=.25);
plt.grid(color='white', linestyle='dashed')
plt.title('Logistic itteration predictions')

plt.subplot(122)
plt.plot(res.resid_dev, 'r-')
plt.grid(color='white', linestyle='dashed')
plt.title('Logistic itteration Residuals');

plt.show()
