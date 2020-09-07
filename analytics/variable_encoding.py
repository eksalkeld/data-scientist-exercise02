# -*- coding: utf-8 -*-
"""
Variable Encoding Approaches
Created on Sat Sep  5 15:02:01 2020

@author: eksalkeld
"""

import pandas as pd

def encode_carrier(df):
        
    df['Carrier']='OtherCarrier'
    df['Carrier'][df['AirCarrier'].str.contains("(?i)DELTA AIRLINES|DELTA AIR LINES")] = "Delta"
    df['Carrier'][df['AirCarrier'].str.contains("(?i)SOUTHWEST AIRLINES|SOUTHWEST AIR LINES")] = "Southwest"
    df['Carrier'][df['AirCarrier'].str.contains("(?i)AMERICAN AIRLINES|AMERICAN AIR LINES")] = "American"
    df['Carrier'][df['AirCarrier'].str.contains("(?i)UNITED AIRLINES|UNITED AIR LINES")] = "United"
    df['Carrier'][df['AirCarrier'].str.contains("(?i)CONTINENTAL AIRLINES|CONTINENTAL AIR LINES")] = "Continental"
    df['Carrier'][df['AirCarrier'].str.contains("(?i)USAIR|US AIRWAYS")] = "USAir"
    df['Carrier'][df['AirCarrier']==''] = "UnknownCarrier"

    return df
    
    
def target_encode_alternative(df,col_name, model_cols):
    
    cat_mean=df.groupby(col_name)['target'].mean()
    
    df.loc[:,col_name+'_tar_enc']=df[col_name].map(cat_mean)
    
    #Transformed column to consider for feature selection
    model_cols.extend([col_name+'_tar_enc'])

    return df, model_cols

import category_encoders
def target_encode(df,col_names, model_cols):
    
    try:
        #Define encoder and what column to evaluate
        ce_target=category_encoders.TargetEncoder(cols=[col_names])
    
        #Train the encoder
        ce_target.fit(df,df['target'])
    
        #Apply the encoder to the data
        df=ce_target.transform(df,df['target'])
        
        #Transformed column to consider for feature selection
        model_cols.extend([col_names])
    
        return df, ce_target, model_cols
    
    #NaN in the target variable throws error
    except ValueError:
        return df, None, model_cols
    
def apply_targetenc(df,ce_target):
    
    try:
        #Apply the encoder to the data
        df=ce_target.transform(df,df['target'])
    
        return df
    
    #If encoder not defined or is not an actual encoder
    except (NameError,AttributeError):
        return df


def woe_encoder(df,col_names, model_cols):
    #Try, ValueError if target has nan outcomes
    try:
        #Define encoder and what column to evaluate
        #Param set to say nan values have WOE=0
        ce_woe=category_encoders.woe.WOEEncoder(cols=[col_names])

        #Train the encoder
        ce_woe.fit(df,df['target'])
    
        #Apply the encoder to the data
        df=ce_woe.transform(df,df['target'])
        
        #Transformed column to consider for feature selection
        model_cols.extend([col_names])

        return df, ce_woe, model_cols
    
    #NaN values in target throw an error
    except ValueError:
        return df, None, model_cols
    
def apply_woe(df,ce_woe):
    
    try:
        #Apply the encoder to the data
        df=ce_woe.transform(df,df['target'])
    
        return df
    
    #If encoder not defined or is not an actual encoder
    except (NameError,AttributeError):
        return df

def freq_encode(df,col_name, model_cols):
    
    #Find the frequency of occurences of each category of a column, relative to the size of the dataframe
    cat_freq=df.groupby(col_name).size()/float(df.shape[0])
    
    #Map the frequencies to their category counterparts, creating a new variable
    df.loc[:,col_name+'_freq_enc']=df[col_name].map(cat_freq)
    
    #Transformed column to consider for feature selection
    model_cols.extend([col_name+'_freq_enc'])
    
    return df, cat_freq, model_cols

def apply_freq(df,cat_freq):
    
    try:
        #Apply the encoder to the data
        df.loc[:,col_name+'_freq_enc']=df[col_name].map(cat_freq)
    
        return df
    
    #If encoder not defined or is not an actual encoder
    except (NameError,AttributeError):
        return df


def dummy_code(df,col_name,model_cols):
    
    #List of original columns
    orig_cols=df.columns
    
    #Create the dummy variables
    df=pd.get_dummies(df,prefix=[col_name],dummy_na=False,columns=[col_name])
    
    #Transformed columns to consider for feature selection
    model_cols.extend(list(set(df.columns)-set(orig_cols)))
    
    return df, model_cols