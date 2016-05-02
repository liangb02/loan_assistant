import cPickle as pickle
import pandas as pd
import numpy as np

from learning.nlp_feature_extraction import title_length, keyword, keyword_lst_generator
from learning.text_analysis import bag_of_words
from learning.model import Model_Pipeline
from flask import Flask, request, redirect, jsonify

app = Flask(__name__, static_url_path="/s")


@app.route('/')
def index():
    return redirect("/s/index.html", code=302)


# my prediction score based on users' input info
@app.route('/api/prediction.json', methods=['POST'])
def prediction():
    '''
    return the calculation score based on the user's input
    and provide the user with suggestions based on their score.
    '''
    name = str(request.form['name'])
    amount = float(request.form['amount'])
    emp_l = float(request.form['emp'])
    dti = float(request.form['dti'])
    msg = str(request.form['message'])

    # Prediction
    df = pd.DataFrame()
    df = df.append({'loan_amnt': amount, 'dti': dti,
                   'emp_length': emp_l, 'title': msg}, ignore_index=True)
    df = process_data(df)
    pro = score_pre(df)

    # Respons
    result = {}
    if pro >= 50.0:
        result['status'] = 'alert'
        result['message'] = 'Congradulations, {0}. You have a great \
                            chance {1}% to get a loan \
                            from Lending Club.'.format(name, pro)
    else:
        result['status'] = 'warning'
        result['message'] = 'Oops, {0}. It seems you might not be able \
                            to get a loan from Lending Club, and the chance\
                            is only {1}%. <br/> You can try: \
                            <br/> 1. Try to limit your Amount [500, 35000], \
                            even better with [7500, 19000]. <br/> \
                            2. You message should be within 33 words, even \
                            better with 2 or 3 words. <br/> \
                            3. Try to avoid words "car" or "business" and use \
                            "personal", "loan", "refinance" and "!" instead. \
                            <br/> Good Luck!'.format(name, pro)

    return (jsonify(**result), 200, {'Content-Type': 'application/json'})


def process_data(df):
    df = title_length(df)

    keyword_lst = [u'!', u'bill', u'business', u'car', u'card',
                   u'cc', u'consolidate', u'consolidation', u'credit',
                   u'debt', u'free', u'freedom', u'home', u'house',
                   u'improvement', u'loan', u'major', u'medical',
                   u'moving', u'pay', u'payoff', u'personal', u'purchase',
                   u'refinance', u'refinancing', u'small', u'vacation',
                   u'wedding']
    df = keyword(df, keyword_lst)
    return df


def score_pre(df):
    fea_lst = ['loan_amnt', 'dti', 'emp_length', u'title_!',
               u'title_bill', u'title_business', u'title_car', u'title_card',
               u'title_cc', u'title_consolidate', u'title_consolidation',
               u'title_credit',
               u'title_debt', u'title_free', u'title_freedom',
               u'title_home', u'title_house',
               u'title_improvement', u'title_length', u'title_loan',
               u'title_major', u'title_medical',
               u'title_moving', u'title_pay', u'title_payoff',
               u'title_personal',
               u'title_purchase', u'title_refinance',
               u'title_refinancing', u'title_small',
               u'title_vacation', u'title_wedding']
    X = df[fea_lst].values
    X_scaled = scaler.transform(X)
    pro = model.predict_proba(X_scaled)[0][0]
    pro = round(pro * 100)
    return pro

if __name__ == '__main__':
    '''
    load the fitted scaler and model from two pickle files
    '''
    with open('data/Lending Club/scaler.pkl') as f:
        scaler = pickle.load(f)

    with open('data/Lending Club/model.pkl') as f:
        model = pickle.load(f)

    app.run(host='0.0.0.0', port=8080, debug=False)
