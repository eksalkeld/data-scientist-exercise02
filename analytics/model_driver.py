# -*- coding: utf-8 -*-
"""
Created on Sun Sep  6 12:58:58 2020

@author: eksalkeld
"""
from data_clean import *
from variable_encoding import *
from feature_selection.py import *
from modeling_fns.py import *
from constants import *

import pandas as pd
#import numpy as np
import xml.etree.ElementTree as ET

import logging
#logging.basicConfig(filename='modeldriverlog.log',format='%(asctime)s %(messages)s',filemode='w')
logger=logging.getLogger('modeling')
logger.setLevel(logging.INFO)

#File handler
hdlr=logging.FileHandler('modeldriverlog.log')

#Console handler
ch=logging.StreamHandler()

#Formatter for file and console
f_formatter=logging.Formatter('%(asctime)s %(levelname)s %(message)s')
c_formatter=logging.Formatter('%(levelname)s %(message)s')
hdlr.setFormatter(f_formatter)
ch.setFormatter(c_formatter)

#Add handlers
logger.addHandler(hdlr)
logger.addHandler(ch)

#Control console output
#pd.set_option('display.max_columns', None)
#np.set_printoptions(threshold=np.nan)

# Load xml file data
tree = ET.parse(AVIATION_FILE)
#Initialize list to hold data
data = []
#https://stackoverflow.com/questions/41795198/more-efficient-conversion-of-xml-file-into-dataframe
#Through first hierarchy
for el in tree.iterfind('./*'):
    #Iterate through each row
    for i in el.iterfind('*'):
        #Access the items in each row and append to the list
        data.append(dict(i.items()))
#Convert list to dataframe
aviation = pd.DataFrame(data)

df=aviation

#Cols for feature selection to consider
model_cols=[]

#Convert columns that need to be numeric into floats
df=convert_numeric(df, cols_to_numeric)

#Create the binary target
df=create_target(df)

#Remove the rows where target is null
df=df[df['target'].notnull()]

#Contend with nas in numerical cols
df=df[df['NumberOfEngines'].notna()]
#df.loc[:,'NumberOfEngines']=df.NumberOfEngines.fillna(0)

#Turn strings of dates to datetimes and also find year/month/day/day of week
df=date_processing(df,'EventDate')
#df=date_processing(df,'PublicationDate')

#Contend with nas in date representations
df=df[df['EventDate_month'].notna()]
df=df[df['EventDate_day'].notna()]
df=df[df['EventDate_dayofweek'].notna()]

#Bin the air carriers
df=encode_carrier(df)

#Dummy encoding
for i in cols_to_dummy:
    df,model_cols=dummy_code(df,i,model_cols)

#Remove one of the categories (the unknown category)
fulldummy=len(model_cols)
model_cols=[x for x in model_cols if not x.endswith('_')]


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
chi_selected,chi_with_dummies=chi2select(df,model_cols)

#Include the numeric variables
model_cols.extend(['NumberOfEngines','EventDate_month','EventDate_day','EventDate_dayofweek'])

#Find predictor columns correlated with each other
corr_cols=find_corr(df[model_cols],cols_to_dummy)

#Vars with lots of missing variables
miss_cols=missingcount(df[model_cols])

#Important variables from random forest
rf_vars=rf_imp(df[model_cols],df.target)

#COMBINE: the yeses of chi and rf, the nos of corr and miss
old_model_cols=model_cols
model_cols=[x for x in model_cols if (x in chi_with_dummies or x in rf_vars) and (x not in corr_cols and x not in miss_cols)]
logger.info('There were %s columns, and feature selection passed %s to the model' % (len(old_model_cols),len(model_cols)))

#Scale the features
dfmodel,scaler=fitscale(df[model_cols],"StandardScaler")

#Add identifier back in
dfmodel=pd.merge(dfmodel, df[['AccidentNumber','EventId']], left_index=True, right_index=True)

#Train, test, val(?) split
if no_val_set:
    X_train, y_train, X_test, y_test=tt_split(dfmodel,df['target'])
else:
    X_train, y_train, X_test, y_test, X_val, y_val=ttv_split(dfmodel,df['target'])
logger.info('The decision to omit a validation set from being created was: %s' % no_val_set)


#Train model
model, chosenmodel, modelc, modelpenalty, modelweight, modelperf=model_train(X_train[model_cols],y_train)
logger.info('The selected reg param: %s' % modelc)
logger.info('The selected model penalty: %s' % modelpenalty)
logger.info('The selected decision for balancing classes: %s' % modelweight)
logger.info('The selected model f1 performance: %s' % modelperf)

#Predictions
train_prob_pred, train_class_pred=model_predict(X_train[model_cols],model)
test_prob_pred, test_class_pred=model_predict(X_test[model_cols],model)

#Merge predictions back into train/test dataframe
X_train['probability']=pd.Series(train_prob_pred)
X_train['class']=pd.Series(train_class_pred)
X_test['probability']=pd.Series(test_prob_pred)
X_test['class']=pd.Series(test_class_pred)

#Coefficients
coefficients=pd.DataFrame(list(zip(model_cols,chosenmodel.coef_[0])),columns=['col_name','coefficient'])
    
#Performance
precision, recall, f1, cm, TN, FP, FN, TP=performance(test_class_pred,y_test)
logger.info('The test set precision: %s' % precision)
logger.info('The test set f1: %s' % f1)
logger.info('The test set recall: %s' % recall)
logger.info('The test set true negatives: %s' % TN)
logger.info('The test set false positives: %s' % FP)
logger.info('The test set false negatives: %s' % FN)
logger.info('The test set true positives: %s' % TP)
    
    
    