import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from nltk.tokenize import sent_tokenize, word_tokenize
import nltk
from collections import Counter
# import matplotlib.pyplot as plt
# import seaborn as sns
import string


'''
This module is to analyze the text features to find useful info
to predict the whether the description will affect Lending Club'
decision to issue a loan to the borrower ('Accepted' or 'Rejected')
NLP_features = rej['Loan Title'] and , accept['title']
1) separate the data into two grounps, i.e. accept and rej
2) find out the most frequent words in each of the two groups
3) apply the key-word features in my model

'''

# process the text features in the data

# 1) lower the words, tokenize the Words
#    remove the stopwords and unicode
# 2) lemmatize the word
#
# 3) build the most frequent words dic for both groups
stop_words = set(stopwords.words('english'))
wordnet = WordNetLemmatizer()


def bag_of_words(content):
    '''
    INPUT: a string like a sentence or paragraph
    'I am a token'
    OUTPUT: a list of words which is tokenized
            removed the unicode, stopwords and
            lemmatized
    '''
    # try:
    if type(content) is unicode:
        content = content.encode('ascii', 'ignore')
    else:
        content = str(content)

    content = string.replace(str(content), '_', ' ')
    tokenize = word_tokenize(str(content).lower())
    words = [w for w in tokenize if (w not in stop_words)]
    lem_words = [wordnet.lemmatize(w) for w in words]

    return lem_words


def build_dic(pd_series):
    '''
    INPUT: a pd_series which contains clean text contents
    OUTPUT: a Counter dict containing the key-words
            and their frequencies in that Series
    The purpose is to find the most frequent word in
    NLP_features and compare the results between two groups
    '''
    N = len(pd_series)
    voc_lst = []
    docs = pd_series.values
    for i in xrange(N):
        voc_lst += bag_of_words(docs[i])
    dic = dict(Counter(voc_lst))
    return dic


def plot_top_keywords(dic, N):
    '''
    INPUT: dict with the count infomation and
           N: the number of top key-words needed
    OUTPUT: plot with the count of the key-word frequencies
            x-axis: Keywords
            y-axis: frequencies of the Keywords
    '''

    s_d = sorted(dic.items(), key=lambda x: x[1], reverse=True)[:N]
    x_l = []
    y = []

    for i in xrange(N):
        x_l.append(s_d[i][0])
        y.append(s_d[i][1])

    x = np.arange(N)
    y = np.array(y)

    plt.figure(figsize=(8, 6))
    plt.bar(x, y, align='center')
    plt.xticks(x, x_l, rotation=30)
    plt.xlabel('Keywords')
    plt.ylabel('Term Frequency')
    plt.title('Keywords and Frequency')
    plt.show()

if __name__ == '__main__':
    df = pd.read_json('../data/Lending Club/acc_rej_data')
    acc = df[df['loan_status'] == 0]
    rej = df[df['loan_status'] == 1]
    d_acc = build_dic(acc['title'])
    plot_top_keywords(d_acc, 10)
    d_rej = build_dic(rej['title'])
    plot_top_keywords(d_rej, 10)
