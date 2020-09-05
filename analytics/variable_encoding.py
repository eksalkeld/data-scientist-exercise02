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


def date_processing(df,col_name,subcomponents=True):
    
    try:
        #String to datetime
        df[col_name]=pd.to_datetime(aviation[col_name])
    
        #If the date components are desired
        if subcomponents:
            #Extract the year
            df[col_name+'_year']=df.apply(lambda x: x[col_name].year, axis=1)
            #Extract the month
            df[col_name+'_month']=df.apply(lambda x: x[col_name].month, axis=1)
            #Extract the day
            df[col_name+'_day']=df.apply(lambda x: x[col_name].day, axis=1)
            #Extract the day of week
            df[col_name+'_dayofweek']=df.apply(lambda x: x[col_name].dayofweek, axis=1)
            
        return df
            
    #If there is an issue with the data provided
    except ValueError:
        return df
