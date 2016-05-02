import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D

from sklearn.cross_validation import train_test_split
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.ensemble.partial_dependence import plot_partial_dependence
from sklearn.ensemble.partial_dependence import partial_dependence
from model import Model_Pipeline
import cPickle as pickle
'''
This module is to EDA the feature importance of the fitted
GradientBoostingClassifier model, as well as the partial
dependence plot. In addition, dump the fitted model and
fitted scaler into pickle for the online real-time prediction.
'''


def plot_feature_importance(feature_lst, fitted_model, N=10):

    '''
    INPUT: feature_lst: a list of features of the model
           fitted_model: the model fitted with the training data
           N: the number of the top features needed in the plot
    OUTPUT: a plot with the feature_names vs. importance
    '''
    fea_imp = fitted_model.feature_importances_
    print fea_imp
    index = np.argsort(fea_imp)[::-1][:N]
    fea = []
    for i in index:
        fea.append(feature_lst[i])
    imp = sorted(fea_imp, reverse=True)[:N]
    fig = plt.figure(figsize=(16, 8))
    plt.bar(range(len(imp)), imp, align='center')
    plt.xticks(range(len(imp)), fea, size='small', rotation=30)
    plt.title('Feature Importance')
    plt.show()

    fig, axs = plot_partial_dependence(fitted_model, X_train, index,
                                       feature_names=feature_lst,
                                       n_jobs = 3, grid_resolution = 50)
    fig.suptitle('Partial dependence of loan request decision on the important features\n'
                 'for the Lending Club dataset')
    plt.subplots_adjust(top=0.9)
    plt.show()


if __name__ == "__main__":
    df = pd.read_json('../data/Lending Club/acc_rej_full_data')
    features = ['loan_amnt', 'dti', 'emp_length']

    title_keyword_list = []
    for col in df.columns:
        if 'title_' in col:
            title_keyword_list.append(col)

    fea_list = features + title_keyword_list
    # fea_list = features+ ['title_length']
    mp = Model_Pipeline(df.copy(), fea_list, 'loan_status',
                        GradientBoostingClassifier(n_estimators=100, learning_rate=1))

    X, y = mp.data_generator()
    X_train, X_test, y_train, y_test, scaler = mp.scale(X, y)
    print mp.cv_score(X_train, y_train)
    mp.fit(X_train, y_train)
    fitted_model = mp.model
    # dump the fitted GradientBoostingClassifier model to the pickle
    # in order to provide online response service.

    # dump the pickle only once for the online evaluation

    # with open('../data/Lending Club/scaler.pkl', 'w') as f:
    #     pickle.dump(scaler, f)
    #
    # with open('../data/Lending Club/model.pkl', 'w') as f:
    #     pickle.dump(fitted_model, f)

    # print type(fitted_model)
    # print type(scaler)
    print fitted_model.score(X_test, y_test)
    plot_feature_importance(fea_list, fitted_model, N=10)
    y_pre = mp.predict(X_test)
    print 'accuracy, precision and recalls are:', mp.score(y_pre, y_test)
