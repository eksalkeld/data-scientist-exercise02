# -*- coding: utf-8 -*-
"""
Created on Sun Sep  6 12:58:58 2020

@author: eksalkeld
"""
from data_clean import *
from variable_encoding import *
from constants import *


#Cols for feature selection to consider
model_cols=[]

#Create the binary target
df=create_target(df)

#Remove the rows where target is null
df=df[df['target'].notnull()]

#Convert columns that need to be numeric into floats
df=convert_numeric(df, cols_to_numeric)

df['NumberOfEngines']=df.NumberOfEngines.fillna(0)

#Turn strings of dates to datetimes and also find year/month/day/day of week
df=date_processing(df,'EventDate')
df=date_processing(df,'PublicationDate')

#Bin the air carriers
df=encode_carrier(df)

#Dummy encoding
for i in cols_to_dummy:
    df,model_cols=dummy_code(df,i,model_cols)

encoders={}

#Find proper encoding for train set
#Frequency encoding
for i in cols_to_freq:
    #Name for storing the encoder
    encoder_name=i+"_encoder"
    #Train encoder and transform data
    df,encoder,model_cols=freq_encode(df,i, model_cols)
    #Store the encoder
    encoders[encoder_name]=encoder
    
#Target encoding
for i in cols_to_target:
    #Name for storing the encoder
    encoder_name=i+"_encoder"
    #Train encoder and transform data
    df,encoder,model_cols=target_encode(df,i, model_cols)
    #Store the encoder
    encoders[encoder_name]=encoder

#WOE encoding
for i in cols_to_woe:
    #Name for storing the encoder
    encoder_name=i+"_encoder"
    #Train encoder and transform data
    df,encoder,model_cols=woe_encoder(df,i, model_cols)
    #Store the encoder
    encoders[encoder_name]=encoder

        
#Feature selection 

#Categorical features associated with response
chi_selected,chi_prefixes=chi2select(df,model_cols)
#PROCESS THE PREFIXES

#Find predictor columns correlated with each other
corr_cols=find_corr(df[model_cols])

#Vars with lots of missing variables
miss_cols=missingcount(df)

#Important variables from random forest
rf_vars=rf_imp(df[model_cols],df.target)

#COMBINE: the yeses of chi and rf, the nos of corr and miss
#model_cols=

#Train, test, val(?) split
if no_val_set:
    X_train, y_train, X_test, y_test=tt_split(df[model_cols],df['target'])
else:
    X_train, y_train, X_test, y_test, X_val, y_val=ttv_split(df[model_cols],df['target'])


#Train model
model, chosenmodel, modelc, modelpenalty, modelweight, modelperf=model_train(X_train,y_train)

#Predictions
train_prob_pred, train_class_pred=model_predict(X_train,model)
test_prob_pred, test_class_pred=model_predict(X_test,model)

#Performance
precision, recall, f1, cm, TN, FP, FN, TP=performance(test_class_pred,y_test)

#User input
    
    
    