# -*- coding: utf-8 -*-
"""
Variable Encoding Approaches
Created on Sat Sep  5 15:02:01 2020

@author: eksalkeld
"""

import pandas as pd
import category_encoders

def encode_carrier(df):
    """
    Bin the air carriers into commonly known airlines, condensing the number of categories
    df:     dataframe with the air carrier column
    return: dataframe with the new column, containing 8 possible carrier values
    """
        
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
    """
    Target encoding not using category_encoders library
    Find the average of the target variable for each level of a categorical variable
    Map the level's target average to each instance of that level 
    Alternative to dummy coding when there are too many levels
    df:         dataframe with the categorical vars to evaluate and the target variable to average
    col_name:   name of categorical variable to transform
    model_cols: list of cols model is considering. Append the name of the new encoded variable to it
    return:     dataframe with transformed var,  value mapping, list of cols for the model
    """
    
    cat_mean=df.groupby(col_name)['target'].mean()
    
    df.loc[:,col_name+'_tar_enc']=df[col_name].map(cat_mean)
    
    #Transformed column to consider for feature selection
    model_cols.extend([col_name+'_tar_enc'])

    return df, cat_mean, model_cols


def target_encode(df,col_names, model_cols):
    """
    Target encoding using category_encoders library
    Find the average of the target variable for each level of a categorical variable
    Fit the encoder to the data and tranform the data with it
    df:         dataframe with the categorical vars to evaluate and the target variable to average
    col_names:  name of categorical variable to transform
    model_cols: list of cols model is considering. Append the name of the new encoded variable to it
    return:     dataframe with transformed var, the trained encoder, list of cols for the model
    """
    
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
    """
    Apply trained encoder to new data
    df:         dataframe with the var to transform
    ce_target:  the trained encoder
    return:     the transformed dataframe
    """
    try:
        #Apply the encoder to the data
        df=ce_target.transform(df,df['target'])
    
        return df
    
    #If encoder not defined or is not an actual encoder
    except (NameError,AttributeError):
        return df


def woe_encoder(df,col_names, model_cols):
    """
    Weight of evidence encoding using category_encoders library
    Fit the encoder to the data and tranform the data with it
    df:         dataframe with the categorical vars to evaluate and the target variable to average
    col_names:  name of categorical variable to transform
    model_cols: list of cols model is considering. Append the name of the new encoded variable to it
    return:     dataframe with transformed var, the trained encoder, list of cols for the model
    """
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
    """
    Apply trained encoder to new data
    df:         dataframe with the var to transform
    ce_woe:     the trained encoder
    return:     the transformed dataframe
    """
    
    try:
        #Apply the encoder to the data
        df=ce_woe.transform(df,df['target'])
    
        return df
    
    #If encoder not defined or is not an actual encoder
    except (NameError,AttributeError):
        return df

def freq_encode(df,col_name, model_cols):
    """
    Frequency encoding - Find the count of the target variable for each level of a categorical variable
    Map the level's target count to each instance of that level 
    Alternative to dummy coding when there are too many levels
    df:         dataframe with the categorical vars to evaluate and the target variable to count
    col_name:   name of categorical variable to transform
    model_cols: list of cols model is considering. Append the name of the new encoded variable to it
    return:     dataframe with transformed var, value mapping, list of cols for the model
    """
    
    #Find the frequency of occurences of each category of a column, relative to the size of the dataframe
    cat_freq=df.groupby(col_name).size()/float(df.shape[0])
    
    #Map the frequencies to their category counterparts, creating a new variable
    df.loc[:,col_name+'_freq_enc']=df[col_name].map(cat_freq)
    
    #Transformed column to consider for feature selection
    model_cols.extend([col_name+'_freq_enc'])
    
    return df, cat_freq, model_cols

def apply_freq(df,cat_freq):
    """
    Apply trained encoder to new data
    df:         dataframe with the var to transform
    cat_freq:   the value mappings for levels
    return:     the transformed dataframe
    """
    
    try:
        #Apply the encoder to the data
        df.loc[:,col_name+'_freq_enc']=df[col_name].map(cat_freq)
    
        return df
    
    #If encoder not defined or is not an actual encoder
    except (NameError,AttributeError):
        return df


def dummy_code(df,col_name,model_cols):
    """
    Create dummy variables for a categorical var
    df:         dataframe with the cat var to transform
    col_name:   name of the var
    model_cols: list of cols the model is considering
    return:     dataframe, list of model cols with the new dummy var names appended
    """
    
    #List of original columns
    orig_cols=df.columns
    
    #Create the dummy variables
    df=pd.get_dummies(df,prefix=[col_name],dummy_na=False,columns=[col_name])
    
    #Transformed columns to consider for feature selection
    model_cols.extend(list(set(df.columns)-set(orig_cols)))
    
    return df, model_cols