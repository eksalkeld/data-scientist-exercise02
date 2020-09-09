# -*- coding: utf-8 -*-
"""
Data Cleaning and Prep Operations
Created on Sun Sep  6 11:57:27 2020

@author: eksalkeld
"""
import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import StandardScaler



def fitscale(df,scaler):
    """
    Scale the data input using either min max, normalizer, or standard scaler
    df:     dataframe where all columns passed in are to be scaled
    scaler: the type of scaling desired (min max vs standard..) 
    return: the normalized dataframe and fit scaler for future use
    """
    if scaler=="MinMaxScaler":
        #Define the scaler
        scaler = MinMaxScaler(feature_range = (0,1))
    elif scaler=="StandardScaler":
        scaler= StandardScaler()
    else:
        scaler=Normalizer()

    #Train the scaler and fit it
    scaler.fit_transform(df)
    
    #Apply the scaler
    #df = scaler.transform(df)
    
    return df, scaler


def apply_scaler(df,scaler):
    """
    If scaler has already been fit, and needs to be applied to the data
    df:     dataframe where all columns passed in are to be scaled
    scaler: scaler that has already been trained (in fitscale)
    return: the scaled dataframe
    """
    
    #Apply the scaler
    df=scaler.transform(df)
    
    return df

def strip_columns(df,suffix_vals):
    """
    Remove columns with a certain suffix, such as IDs that should not be model input
    df:             The dataframe with the column names to search for the suffix that will be ommitted
    suffix_values:  the value of the suffix of columns to be removed. If suffix is not right type, error handling
    return:         return the list of subset column names
    """
    
    #Suffix_vals should be string or tuple
    try:
        #Find the column names that don't have the specified suffix(es)
        col_list=[x for x in df.columns if not x.endswith(suffix_vals)] 
    
        #Return the list of column names for subsetting
        return col_list
    
    #If suffix_vals not the right type
    except TypeError:
        return df.columns
    
def date_processing(df,col_name,subcomponents=True):
    """
    Turn string to date time and if desired strip off year/month/day/day of week
    df:             dataframe with the column to transform into date time
    col_name:       the name of the column from which to create the date time
    subcomponents:  boolean as to whether to strip off the year/month/day into their own columns, defaults to true
    return:         dataframe with the new date columns
    """
    
    try:
        dfdate=df.copy()
        #String to datetime
        dfdate.loc[:,col_name]=pd.to_datetime(df[col_name])
    
        #If the date components are desired
        if subcomponents:
            #Extract the year
            dfdate.loc[:,col_name+'_year']=dfdate.apply(lambda x: x[col_name].year, axis=1)
            #Extract the month
            dfdate.loc[:,col_name+'_month']=dfdate.apply(lambda x: x[col_name].month, axis=1)
            #Extract the day
            dfdate.loc[:,col_name+'_day']=dfdate.apply(lambda x: x[col_name].day, axis=1)
            #Extract the day of week
            dfdate.loc[:,col_name+'_dayofweek']=dfdate.apply(lambda x: x[col_name].dayofweek, axis=1)
            
        return dfdate
            
    #If there is an issue with the data provided
    except ValueError:
        return df
    
def create_target(df):
    """
    Create a binary 1/0 target to use for modeling
    df:     Dataframe with the columns needed to create the target
    return: dataframe with the 1/0 target variable
    """
    
    #Find the number of fatal, minor, and serious injuries for each flight
    #Sum of any kind of bodily harm from a flight
    df['Injured']=df[['TotalFatalInjuries', 'TotalMinorInjuries', 'TotalSeriousInjuries']].sum(axis=1)
    
    #Find the rows where no counts of injured, fatalities, or uninjured were reported
    #We will want to omit them so not to make assumptions on the outcome of the flight
    df['ReportsNull']=df[['TotalFatalInjuries','TotalSeriousInjuries', 'TotalMinorInjuries', 'TotalUninjured']].isnull().all(1)
    
    #Create 1/0 target variable
    #1 if there was any kind of injury, fatal or otherwise
    #0 if there all reports were uninjured
    #NaN if there were not any reports - these will later be dropped
    df['target']=df.apply(lambda x: np.nan if x['ReportsNull']==True else (1 if x['Injured']>0 else 0) ,axis=1)

    return df
    
    
    
    
def convert_numeric(df, cols_to_numeric):
    """
    Convert string columns into numeric columns
    df:                 dataframe with the columns to transform into numerics
    cols_to_numeric:    list of columns that should be made numeric
    return:             dataframe with the columns now numeric
    """
    
    #loop through each column that needs to be converted and transform to numeric
    for i in cols_to_numeric:
        #df[i]=pd.to_numeric(df[i],errors='coerce')
        df.loc[:,i] = pd.to_numeric(df[i],errors='coerce')
        
    return df
    

    
    