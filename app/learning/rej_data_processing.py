import pandas as pd
import numpy as np
from model import Model_Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from text_analysis import bag_of_words
from nlp_feature_extraction import title_length, keyword


'''
This module is to preposs the rejection data in the speadsheets
RejectStatsA.csv
RejectStatsB.csv

LoanStats3a_securev1
LoanStats3b_securev1
from 2007 to 2013 as they provide suffient text information in 'title'.

The purpose is to compare with the lending club accepted data and rejected data with
the labels (0) as accepted loan requests and (1) as the rejected loan requests.

Part I: To process the rejection data:

1) 'Amount Requested': rename the column name to 'loan_amnt' keep
consistent with the columns with the accepted data.

2) 'Debt-To-Income Ratio': strip the '%' and convert to float,
                           rename the column to 'dti'.

3) 'Employment Length': clean the data as int,
    and rename the column to 'emp_length'

4) create the all label columns 'loan_status' as '1'
feature_set = ['loan_amnt','dti','emp_length', 'title', 'loan_status']
'''


def emp_length_clean(st):
    if st == '10+ years':
        return 10
    elif st == '< 1 year':
        return 0.5
    elif st == 'n/a':
        return np.nan
    elif type(st) == float:
        return np.nan
    else:
        return int(st.strip()[0])

# read in data and merge as one rejection dataframe 'rej'

rejA = pd.read_csv('../data/Lending Club/RejectStatsA.csv',  skiprows=[0])
rejB = pd.read_csv('../data/Lending Club/RejectStatsB.csv',  skiprows=[0])

# Take the data of 2013 only in rejB
rejB = rejB[rejB['Application Date'].apply(lambda x: x[:4] == '2013')]
# rejD = pd.read_csv('../data/Lending Club/RejectStatsD.csv',  skiprows=[0])

frames = [rejA, rejB]
rej = pd.concat(frames)
rej = rej.reset_index()

# rename the 'Amount Requested' as 'loan_amnt'
# rename the 'Debt-To-Income' Ratio' as 'dti'
# rename the 'Employment Length' as 'emp_length'
# also rename the 'Loan Title' as 'title' for future text analysis

# 1) rename columns
rej.rename(columns={'Amount Requested': 'loan_amnt',
                    'Debt-To-Income Ratio': 'dti',
                    'Employment Length': 'emp_length',
                    'Loan Title': 'title'}, inplace=True)


# 2) 'Debt-To-Income Ratio': strip the '%' and convert to float.
rej['dti'] = rej['dti'].apply(lambda x: float(x.strip('%')))

# 3) convert the emp_length column to int type and
# deal with the special ones '<1 year' and '+10 years'
rej['emp_length'] = rej['emp_length'].apply(emp_length_clean)

# 4) add the label column 'loan_status' as '1'
rej['loan_status'] = 1

'''
Part II: is to read the data from the spreadsheet in
order to merge with the rejection data and further to process the
data for the model.
'''

# Read two speadsheets as my final data
df2011 = pd.read_csv('../data/Lending Club/LoanStats3a_securev1.csv', skiprows=[0])
df2012 = pd.read_csv('../data/Lending Club/LoanStats3b_securev1.csv', skiprows=[0])

# merge the dataframes as one and select the columns
frames = [df2011, df2012]
accept = pd.concat(frames)
accept = accept.reset_index()
accept = accept[['loan_amnt', 'dti', 'emp_length', 'title', 'loan_status']]

# proess the 'emp_length' and 'loan_status' columns
accept['emp_length'] = accept['emp_length'].apply(emp_length_clean)

accept['loan_status'] = 0


# As the dataset is skewed and we got big dataset, undersample the rej data.
# combine two groups as a new one.

sample_rej = rej[['loan_amnt', 'dti', 'emp_length', 'title', 'loan_status']].sample(n=len(accept), replace=False)
data_set = pd.concat([accept, sample_rej])
data_set = data_set.reset_index()

# save the data set as the json to process, only run once
# data_set.to_json('../data/Lending Club/acc_rej_data')

# The following will create a new dataset with NLP_features
