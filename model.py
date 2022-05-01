import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('train.csv')
dataset['type'].fillna(int(dataset['type'].mean()), inplace=True)
dataset.columns = ['ID','type','time','dayofweek','models','params','queuelen','trails','duration']
dataset.drop(['ID'],axis=1,inplace=True)
dataset['params'].fillna((dataset['params'].mean()), inplace=True)

X = dataset.drop(['duration'],axis=1)
Y = dataset.duration

from sklearn.model_selection import train_test_split
x_ttrain, x_ttest, y_ttrain, y_ttest = train_test_split(X, Y, random_state = True, test_size = 0.25) 

from sklearn.ensemble import RandomForestClassifier
modRFC = RandomForestClassifier()
modRFC.fit(x_ttrain,y_ttrain)

import pickle
filename = 'rfcmodel.pkl'
pickle.dump(modRFC, open(filename, 'wb'))