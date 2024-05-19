import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

train_data = pd.read_csv('train.csv')

target = train_data['label']
data = train_data.drop(['id','label'], axis = 1)

columns = train_data.columns

# Feature Scaling

from sklearn.preprocessing import StandardScaler

columns = data.columns
ss = StandardScaler()
data = ss.fit_transform(data)
data = pd.DataFrame(data, columns = columns)

from sklearn.feature_selection import SelectKBest, chi2

from sklearn.feature_selection import SelectKBest, f_classif

kbest = SelectKBest(f_classif, k=10)
train = kbest.fit(data, target)

cols = kbest.get_support(indices=True)
new_train = data.iloc[:,cols]

# Applying Gradient Boosting model to the data

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

gbc = GradientBoostingClassifier()

param_grid_gbc = {'n_estimators' : [80, 100,125,150], 'learning_rate': [0.001, 0.01, 0.1, 1]}
grid_gbc = GridSearchCV(gbc, param_grid = param_grid_gbc, cv=5, verbose=1, n_jobs=-1)
grid_gbc.fit(new_train, target)

print(grid_gbc.best_estimator_)
print(grid_gbc.best_params_)
print(grid_gbc.best_score_)

y_pred_gbc = grid_gbc.best_estimator_.predict(new_train)

pickle.dump(grid_gbc, open('model.pkl','wb'))

model = pickle.load(open('model.pkl','rb'))