'''
This module is to create nlp features and process the features in order to
being added to the lending club classifier model.
'''
import pandas as pd
import numpy as np
from text_analysis import bag_of_words, build_dic
import re


def title_length(df):
    '''
    This function is to create a new feature column 'title_length'
    and add the new column to the existing dataframe.
    INPUT: the original dataframe
    OUTPUT: the dataframe with the new feature column
    '''
    df['title_length'] = df['title'].apply(lambda x: len(re.split(' |_', x.encode('ascii', 'ignore'))) if not x is None else 0)
    return df


def keyword_lst_generator(df, column_name, N, label):
    '''
    INPUT: dataframe
           column_name: name of the columns needed NLP analysis
           N: the number of top most frequent appearance keywords
           label
    OUTPUT: a set of keywords of top N most frequent
    '''
    acc = df[df[label] == 0]
    rej = df[df[label] == 1]
    d_acc = build_dic(acc[column_name])
    d_rej = build_dic(rej[column_name])
    s_d_acc = sorted(d_acc.items(), key=lambda x: x[1], reverse=True)[:N]
    s_d_rej = sorted(d_rej.items(), key=lambda x: x[1], reverse=True)[:N]
    word_set = set()
    for word in s_d_acc:
        word_set.add(word[0])
    for word in s_d_rej:
        word_set.add(word[0])
    return word_set


def keyword(df, keyword_lst):
    '''
    This function is to create new feature columns using the keyword
    in the keyword_lst.
    INPUT: dataframe and the keyword list
    OUTPUT: dataframe with the new keyword columns
            if the title contain the keyword, set as '1'
            otherwise set '0'
    '''
    df['key_lst'] = df['title'].apply(bag_of_words)
    for key in keyword_lst:
        df['title_' + key] = df['key_lst'].apply(lambda x: key in x).astype(int)
    return df

if __name__ == '__main__':
    df = pd.read_json('../data/Lending Club/acc_rej_data')
    df = title_length(df)
    keyword_lst = keyword_lst_generator(df, 'title', 20, 'loan_status')
    df = keyword(df, keyword_lst)
    # print df.head()
    # df.to_json('../data/Lending Club/acc_rej_full_data')
