import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score, precision_score


class Model_Pipeline(object):
    def __init__(self, df, feature_lst, label, model=LogisticRegression()):
        '''
        INPUT: "feature_lst" is the list of features which will be used
                in the model tp predict the result
               "df" is the dataframe containing all the columns
               "label" is the column contain the '0' or '1'
                in the project, '1' means rejected and '0' means accepted
               "model" is the machine learning model to be used in the class
        '''
        self.df = df
        self.features = feature_lst
        self.data_frame = df[feature_lst]
        self.label = label
        self.model = model

    def data_generator(self):
        '''
        INPUT: dataframe, features and label
        OUTPUT: X, y

        The Method is to extract data and prepossing data
        1) fill in the missing values with mean values
        2) extract data and labels from dataframe
        '''
        self.data_frame.fillna(self.data_frame.mean(), inplace=True)
        X = self.data_frame.values
        y = self.df[self.label].values
        return X, y

    def scale(self, X, y):
        '''
        INPUT: X, y
        OUTPUT: X_train, X_test, y_train, y_test (all scaled)
                and the fitted scaler using the training data

        The Method is to split the data into trainging data and
        testing data and the fitted scaler

        1) Split the data into training and testing data
        2) scale both training data and test data use the training data set
        3) fitted the scaler using the training data
        '''
        X_train, X_test, y_train, y_test = train_test_split(X, y)
        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)
        return X_train, X_test, y_train, y_test, scaler

    def fit(self, X_train, y_train):
        '''
        INPUT: Scaled Training X and Scaled Training y,
               and model(e,g,Logisticregressor())
        OUTPUT: None
        This method is to fit the data with the training data
        '''

        self.model.fit(X_train, y_train)

    def cv_score(self, X_train, y_train):
        '''
        INPUT: training X and y
        OUTPUT: means cross_validation score on training data
        NOTE: calculate the cv score before fitting the model
        '''
        scores = cross_val_score(self.model, X_train, y_train)
        return np.mean(scores)

    def predict(self, X_test):
        '''
        INPUT: fitted model and X_test
        OUTPUT: predict values y_pred
        NOTE: call the predict method after fitting the model
        '''
        y_pred = self.model.predict(X_test)
        return y_pred

    def score(self, y_pre, y_test):
        '''
        INPUT: y_pred, y_true
        OUTPUT: precision_score, accuracy_score, recall_score
        '''
        recall = recall_score(y_test, y_pre)
        accuracy = accuracy_score(y_test, y_pre)
        precision = precision_score(y_test, y_pre)
        return accuracy, precision, recall
