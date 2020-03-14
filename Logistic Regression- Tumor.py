# -*- coding: utf-8 -*-
"""
Created on Sat Mar 14 17:24:10 2020

@author: Ishan
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data=pd.read_csv(r'C:\Users\Ishan\Documents\Python Scripts\Datasets\Tumor- Malignant or benign.csv')
print(data.head)

sns.jointplot('radius_mean','texture_mean',data=data)
sns.heatmap(data.corr())

data.isnull().sum()

X=data[['radius_worst','texture_worst','perimeter_worst','area_worst','smoothness_worst','compactness_worst','concavity_worst','concave points_worst','symmetry_worst','fractal_dimension_worst']]
y=data[['diagnosis']]
X.head()
y.head()

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=101)

from sklearn.linear_model import LogisticRegression
logmodel=LogisticRegression()
logmodel.fit(X_train,y_train)
y_pred=logmodel.predict(X_test)

from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred))


