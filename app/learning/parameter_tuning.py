from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.cross_validation import train_test_split, cross_val_score
from sklearn.grid_search import GridSearchCV
import pandas as pd
import numpy as np
from model import Model_Pipeline

'''
This module will tune the parameters of five potential classifiers
with the metrix of build-in function accuracy of cross_val_score

1) LogisticRegression
2) RandomForestClassifier
3) GradientBoostingClassifier
4) AdaBoostClassifier

'''

# logistic_grid = {'penalty': ['l2', 'l1'],
#                  'tol': [0.00001, 0.0001, 0.001],
#                  'C': [0.1, 1, 10],
#                  'max_iter': [50, 100, 1000]}

# random_forest_grid = {'max_depth': [3, None],
#                       'max_features': ['sqrt', 'log2', None],
#                       'min_samples_split': [1, 2, 4],
#                       'min_samples_leaf': [1, 2, 4],
#                       'bootstrap': [True, False],
#                       'n_estimators': [10, 20, 40],
#                       'random_state': [1, None]}
#
# gradient_boost_grid = {'learning_rate': [0.05, 0.1, 0.3],
#                        'max_depth': [1, 3, 5],
#                        'min_samples_leaf': [1, 2, 4],
#                        'max_features': ['sqrt', 'log2', None],
#                        'n_estimators': [50, 100, 200]}
#
# ada_boost_grid = {'learning_rate': [0.5, 1, 3],
#                   'algorithm': ['SAMME.R', 'SAMME'],
#                   'n_estimators': [25, 50, 100]}

# svm_grid = {'C': [0.1, 1, 10],
#             'kernel': ['linear', 'rbf', 'sigmoid'],
#             'degree': [1, 3, 5],
#             'gamma': ['auto', 0.01, 0.1],
#             'tol': [0.0001, 0.001, 0.01]}

# test case on small gridsearch parameters and deploy the whole parameter-set
# on the AWS

logistic_grid = {'C': [0.1, 1, 10]}

random_forest_grid = {'min_samples_leaf': [1, 3],
                      'n_estimators': [10, 20]}

gradient_boost_grid = {'learning_rate': [0.1, 1],
                       'n_estimators': [50, 100]}

ada_boost_grid = {'learning_rate': [0.1, 1],
                  'n_estimators': [50, 100]}

model_lst = [LogisticRegression(),
             RandomForestClassifier(),
             GradientBoostingClassifier(),
             AdaBoostClassifier()]


grid_lst = [logistic_grid,
            random_forest_grid,
            gradient_boost_grid,
            ada_boost_grid]


tuning_data = pd.read_json('../data/Lending Club/acc_rej_full_data')
features = ['loan_amnt', 'dti', 'emp_length']

title_keyword_list = []
for col in tuning_data.columns:
    if 'title_' in col:
        title_keyword_list.append(col)

feature_lst = features + title_keyword_list

model = Model_Pipeline(tuning_data.copy(), feature_lst, 'loan_status')
print tuning_data[feature_lst].info()

X, y = model.data_generator()
X_train, X_test, y_train, y_test, scaler = model.scale(X, y)
for i in xrange(len(grid_lst)):
    gs = GridSearchCV(model_lst[i],
                      grid_lst[i],
                      n_jobs=-1,
                      verbose=True,
                      scoring='accuracy')
    gs.fit(X_train, y_train)
    print 'Model: ', model_lst[i]
    print 'Best Parameters: ', gs.best_params_
    print 'Best Score: ', gs.best_score_
