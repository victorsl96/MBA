import pandas as pd
import numpy as np

from sklearn.model_selection import cross_validate
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

import matplotlib.pyplot as plt


data = pd.read_csv('NBA_Players.csv')

data.drop(columns=[' URL'], inplace=True)
# data.drop(data.loc[data['SALARY'] == 'Not signed'].index, inplace=True)

correlation = data.corr()
correlation.plot()
# print(type(correlation))
# print(data.describe())
# print(train_test_split(data))
# cm = plt.get_cmap('viridis')
