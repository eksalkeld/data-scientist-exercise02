# -*- coding: utf-8 -*-
"""
Created on Wed Sep  2 17:56:54 2020

@author: eksalkeld
"""



import json
import pandas as pd
import xml.etree.ElementTree as ET
from pandas.io.json import json_normalize
import glob, os
import sys

from constants.py import *

pd.set_option('display.max_columns', None)
import numpy as np
#np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(threshold=np.nan)

# Load xml file data
tree = ET.parse(AVIATION_FILE)
#Initialize list to hold data
data = []
#https://stackoverflow.com/questions/41795198/more-efficient-conversion-of-xml-file-into-dataframe
for el in tree.iterfind('./*'):
    for i in el.iterfind('*'):
        data.append(dict(i.items()))
#Convert list to dataframe
aviation = pd.DataFrame(data)



JSON_FILE='C://Users//eksalkeld//Documents//GitHub//data-scientist-exercise02//data//NarrativeData_000.json'
with open(JSON_FILE, 'r') as f:
    jsondata = json.load(f)

jsondf=json_normalize(jsondata['data'])




os.chdir(JSON_FOLDER)
json_files=[]
for file in glob.glob("*.json"):
    json_files.append(file)
    
    
    


####################################################################DATA CLEANING
#Convert date time
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

#Convert string to float
cols_to_numeric=['Latitude','Longitude','TotalFatalInjuries', 'TotalMinorInjuries', 'TotalSeriousInjuries','TotalUninjured']

for i in cols_to_numeric:
    aviation[i]=pd.to_numeric(aviation[i],errors='coerce')
    

aviation=date_processing(aviation,'EventDate')
aviation=date_processing(aviation,'PublicationDate')
    
##################################################################################
#########################OUTCOME REPRESENTATIONS##################################
outcome_types=['TotalFatalInjuries','TotalSeriousInjuries', 'TotalMinorInjuries', 'TotalUninjured']

#Additional Representations
aviation['TotalCount']=aviation[outcome_types].sum(axis=1)
aviation['NonFatalInjured']=aviation[['TotalMinorInjuries', 'TotalSeriousInjuries']].sum(axis=1)
aviation['Injured']=aviation[['TotalFatalInjuries', 'TotalMinorInjuries', 'TotalSeriousInjuries']].sum(axis=1)

#Extend for new representation
outcome_types.extend(['TotalCount','NonFatalInjured','Injured'])

#Ratios for each outcome
for i in outcome_types:
    aviation[i+'_Ratio']=aviation[i].div(aviation['TotalCount'])
    

############################Build ground truth########################################
#Sudo target for binary prediction
#1= value for total fatal, minor, serious
#0= value for uninjured
#nan if none filled out

aviation['ReportsNull']=aviation[['TotalFatalInjuries','TotalSeriousInjuries', 'TotalMinorInjuries', 'TotalUninjured']].isnull().all(1)
aviation['target']=aviation.apply(lambda x: np.nan if x['ReportsNull']==True else (1 if x['Injured']>0 else 0) ,axis=1)

###################################################################################
#########################DATE EXPLORATION##########################################
#Time between event and a publication
aviation['ReportingTime']=aviation['PublicationDate']-aviation['EventDate']

#Yearly sum of injuries
injured_a_year=aviation[['Injured','EventDate_year']].groupby('EventDate_year').sum().reset_index()
#Yearly sum of all passangers accounted for in publication
total_a_year=aviation[['TotalCount','EventDate_year']].groupby('EventDate_year').sum().reset_index()

#Join yearly measures
yearly_breakdown=injured_a_year.merge(total_a_year,how='inner',on='EventDate_year')

#Ratio of all people in publications that were put at risk by some injury, fatal or not
yearly_breakdown['RiskRatio']=yearly_breakdown['Injured'].div(yearly_breakdown['TotalCount'])
###################################################################################

import seaborn as sns
import matplotlib.pyplot as plt

#Remove data before 1980
plotyear=aviation[aviation['EventDate_year']>=1980]

#RiskRatio
sns.set_style('whitegrid')
snsplot=sns.lmplot('EventDate_year','TotalFatalInjuries_Ratio',data=plotyear,hue='AircraftCategory',fit_reg=False,size=5,aspect=1)
fig=snsplot.fig
plt.title('Risk over Time',fontsize=15)
plt.xlabel('Year',fontsize=10)
plt.ylabel('Fatal Injuries to Total Passangers',fontsize=10)
fig.savefig('RiskTime.png',bbox_inches='tight')
#snsplot.savefig('ELIZA.png')

snsplot=sns.lmplot('EventDate_year','RiskRatio', data=yearly_breakdown,fit_reg=False)
fig=snsplot.fig
plt.title('Risk over Time',fontsize=15)
plt.xlabel('Year',fontsize=10)
plt.ylabel('All Injuries to Total Passangers',fontsize=10)
fig.savefig('RiskTime2.png',bbox_inches='tight')

#########################PLANE SPECS EXPLORATION##########################################

plane_cols=['AirCarrier', 'AircraftCategory', 'AircraftDamage','AmateurBuilt', 'EngineType', 'EventDate',\
 'Make', 'Model', 'NumberOfEngines','PurposeOfFlight', 'RegistrationNumber']

for i in plane_cols:
    print(i+": "+str(len(aviation[i].unique())))


#JUST THE AIRPLANES
planes=aviation[aviation['AircraftCategory']=='Airplane']
#Yearly sum of injuries
planes_injured_a_year=planes[['Injured','EventDate_year']].groupby('EventDate_year').sum().reset_index()
#Yearly sum of all passangers accounted for in publication
planes_total_a_year=planes[['TotalCount','EventDate_year']].groupby('EventDate_year').sum().reset_index()

#Join yearly measures
planes_yearly_breakdown=planes_injured_a_year.merge(planes_total_a_year,how='inner',on='EventDate_year')

#Ratio of all people in publications that were put at risk by some injury, fatal or not
planes_yearly_breakdown['RiskRatio']=planes_yearly_breakdown['Injured'].div(planes_yearly_breakdown['TotalCount'])

snsplot=sns.lmplot('EventDate_year','RiskRatio', data=planes_yearly_breakdown,fit_reg=True)
fig=snsplot.fig
plt.title('Risk over Time',fontsize=15)
plt.xlabel('Year',fontsize=10)
plt.ylabel('All Injuries to Total Passangers',fontsize=10)
fig.savefig('AirplaneRiskTime2.png',bbox_inches='tight')

snsplot=sns.lmplot('EventDate_year','Injured', data=planes,fit_reg=False,hue='EventDate_dayofweek')
fig=snsplot.fig
plt.title('Injuries from Planes over Time',fontsize=15)
plt.xlabel('Year',fontsize=10)
plt.ylabel('Injury counts of Passangers',fontsize=10)
fig.savefig('AirplaneRiskTime2.png',bbox_inches='tight')



#########################WEATHER EXPLORATION##########################################

pd.crosstab(aviation.AircraftDamage,aviation.WeatherCondition)

#IMC is lowest in the summer and VMC higher in the summer -- winter weather worse to travel in?
#UNK is unknown? not showing up in googling weather codes...
pd.crosstab(aviation.EventDate_month,aviation.WeatherCondition)

pd.crosstab(aviation.WeatherCondition,aviation.target)
#VMC non injury rate= 0.57066
39190/(39190+29484)
#IMC non injury rate= 0.228284
1272/(1272+4300)
#########################LOCATION EXPLORATION##########################################
['Country','Latitude','Longitude','Location']

aviation.groupby('Country').EventId.count().reset_index().sort_values('EventId',ascending=False)
aviation.groupby('Location').EventId.count().reset_index().sort_values('EventId',ascending=False)

pd.crosstab(aviation.Location,aviation.target)

pd.crosstab(aviation.Country,aviation.target)




###############################GROUP BY EXPLORATION##################################
#Group bys
    
aviation.groupby('WeatherCondition').count()

aviation.groupby('AmateurBuilt').count()
    
aviation.groupby('PurposeOfFlight').count()

aviation[['AccidentNumber','ReportStatus']].groupby('ReportStatus').count().reset_index().sort_values('AccidentNumber',ascending=False)
    
aviation[['AccidentNumber','EventId']].groupby('EventId').count().reset_index().sort_values('AccidentNumber',ascending=False)

aviation[['AccidentNumber','InjurySeverity']].groupby('InjurySeverity').count().reset_index().sort_values('AccidentNumber',ascending=False)
    
aviation[aviation['InjurySeverity']=='Fatal(12)']

aviation[aviation['EventId']=='20001214X45071']


aviation[['AccidentNumber','TotalFatalInjuries']].groupby('TotalFatalInjuries').count().reset_index().sort_values('AccidentNumber',ascending=False)


pd.crosstab(aviation['year'],aviation['PurposeOfFlight'])


    
aviation['TotalCount']=aviation[['TotalFatalInjuries', 'TotalMinorInjuries', 'TotalSeriousInjuries','TotalUninjured']].sum(axis=1)
aviation0=aviation[aviation['TotalCount']==0]