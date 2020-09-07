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
    
    #Apply the scaler
    df=scaler.transform(df)
    
    return df

def strip_columns(df,suffix_vals):
    
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
    
    #loop through each column that needs to be converted and transform to numeric
    for i in cols_to_numeric:
        #df[i]=pd.to_numeric(df[i],errors='coerce')
        df.loc[:,i] = pd.to_numeric(df[i],errors='coerce')
        
    return df
    

    
    