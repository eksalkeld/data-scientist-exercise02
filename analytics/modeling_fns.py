# -*- coding: utf-8 -*-
"""
Created on Sun Sep  6 15:57:19 2020

@author: eksalkeld
"""

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
import numpy as np
from constants import *

def pd_to_np(df,model_cols):
    
    X=np.array(df[model_cols])
    y=np.array(df['target'])
    
    return X,y


def ttv_split(X,y):
    
    #Adjust the ratios for how the algorithm understands them
    test_size1=1-train_proportion
    test_size2=test_size1-val_proportion
    
    #Pull off the train set
    X_train, X_hold, y_train, y_hold = train_test_split(X, y,stratify=y,test_size=test_size1,random_state=seed)
    
    #Pull off the test and val sets
    X_test, X_val, y_test, y_val = train_test_split(X_hold, y_hold,stratify=y_hold,test_size=test_size2,random_state=seed)
    
    return X_train, y_train, X_test, y_test, X_val, y_val

def tt_split(X,y):
    
    #Adjust the ratios for how the algorithm understands them
    test_size1=1-train_proportion
    
    #Pull off the train set
    X_train, X_test, y_train, y_test = train_test_split(X, y,stratify=y,test_size=test_size1,random_state=seed)
    
    return X_train, y_train, X_test, y_test


def model_train(X,y):
    
    penalty = ['l1', 'l2']
    classweight=[None,'balanced']
    c = [0.0001, 0.001, 0.01, 1, 100]
    
    
    penalty = ['l2']
    classweight=['balanced']
    c = [ 1]
    
    
    #Logistic regression
    lr =LogisticRegression(random_state=seed)
    #Hyper params for the grid to evaluate
    random_grid = dict(C=c, penalty=penalty,class_weight=classweight)
    #Define the grid search
    cvlr = GridSearchCV(estimator=lr,param_grid=random_grid, scoring='f1', cv= kfold)
    #Fit to the data
    model=cvlr.fit(X,y)
    #Obtain the selected parameters
    chosenmodel=model.best_estimator_
    modelc=model.best_params_['C']
    modelpenalty=model.best_params_['penalty']
    modelweight=model.best_params_['class_weight']
    #Performance
    modelperf=model.best_score_
    
    return model, chosenmodel, modelc, modelpenalty, modelweight, modelperf


def model_predict(X,model):
    
    #Predict probability
    prob_pred=model.predict_proba(X)[:,1]
    
    class_pred=model.predict(X)
    
    return prob_pred, class_pred

def performance(class_pred,y):
    
    from sklearn.metrics import classification_report, confusion_matrix, roc_curve
    
    scores=classification_report(y,class_pred,output_dict=True)['1.0']
    precision=scores['precision']
    recall=scores['recall']
    f1=scores['f1-score']
    
    cm=confusion_matrix(y,class_pred)
    TN=cm[0][0]
    FP=cm[0][1]
    FN=cm[1][0]
    TP=cm[1][1]
        
    return precision, recall, f1, cm, TN, FP, FN, TP
