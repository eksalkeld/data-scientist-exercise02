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

#Convert columns that need to be numeric into floats
df=convert_numeric(df, cols_to_numeric)

#Turn strings of dates to datetimes and also find year/month/day/day of week
df=date_processing(df,'EventDate')
df=date_processing(df,'PublicationDate')
    
#Create the binary target
df=create_target(df)

#Bin the air carriers
df=encode_carrier(df)

#Dummy encoding
for i in cols_to_dummy:
    df,model_cols=dummy_code(df,i,model_cols)

#Train, test, val split
train=df
test=df
val=df

#Find proper encoding for train set
#Frequency encoding
for i in cols_to_freq:
    encoder_name=i+"_encoder"
    train,encoder_name,model_cols=freq_encode(train,i, model_cols)
    
#Target encoding
for i in cols_to_target:
    encoder_name=i+"_encoder"
    train,encoder_name,model_cols=target_encode(train,i, model_cols)

#WOE encoding
for i in cols_to_woe:
    encoder_name=i+"_encoder"
    train,encoder_name,model_cols=woe_encoder(train,i, model_cols)
    
#Process test and validation sets with the encoding
for j in [test,val]:
    
    #Frequency encoding
    for i in cols_to_freq:
        encoder_name=i+"_encoder"
        j=apply_freq(j,encoder_name)
    
    #Target encoding
    for i in cols_to_target:
        encoder_name=i+"_encoder"
        j=apply_targetenc(j,encoder_name)

    #WOE encoding
    for i in cols_to_woe:
        encoder_name=i+"_encoder"
        j=apply_woe(j,encoder_name)
        
#Feature selection on train set


#Train model
        

#Predictions
        
        

#Performance
    

#User input
    
    
    