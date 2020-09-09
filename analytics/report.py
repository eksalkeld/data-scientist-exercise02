# -*- coding: utf-8 -*-
"""
Created on Tue Sep  8 11:22:07 2020

@author: eksalkeld
"""
from constants import *

import pandas as pd
import xml.etree.ElementTree as ET
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

#Control console output
pd.set_option('display.max_columns', None)
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

df=aviation

#Confirm each column is an object
df.dtypes

missing_count=[]
for i in df.columns:
        missing_count.append([i,df[df[i]==''].shape[0]])
pd.DataFrame(missing_count,columns=['name','count']).sort_values('count')
#Number of missing values in each column
for i in df.columns:
    print(str(i)+": "+str(df[df[i]==''].shape[0]))
    
#Unique events to unique accidents
len(df.EventId.unique()) #76133
len(df.AccidentNumber.unique()) #77257

#Greatest number of events per accident
df.groupby('EventId').size().reset_index().rename(columns={0:'Count'}).sort_values('Count',ascending=False)
#df.groupby('EventId').AccidentNumber.count().reset_index().rename(columns={'AccidentNumber':'Count'}).sort_values('Count',ascending=False)
#Max is 3 events

#Number of unique values per column
for i in df.columns:
    print(str(i)+": "+str(len(df[i].unique())))
    
#Total count per year
df['TotalCount']=df[['TotalFatalInjuries','TotalMinorInjuries','TotalSeriousInjuries','TotalUninjured']].sum(axis=1)
graph=df.groupby('EventDate_year').TotalCount.sum().reset_index()
plt.scatter('EventDate_year','TotalCount', data=graph)


#Create a 1/0 of if there were injuries fatal or otherwise
df=convert_numeric(df, ['TotalFatalInjuries','TotalMinorInjuries','TotalSeriousInjuries','TotalUninjured'])
df=create_target(df)
    
#Turn strings of dates to datetimes and also find year/month/day/day of week
df=date_processing(df,'EventDate')

'''
##############################################Graph One#################################
##############################################Too complicated for one page description#################################
#How many reports per year
yearly_report=df.groupby('EventDate_year').AccidentNumber.count().reset_index().rename(columns={'AccidentNumber':'Count'})
   
#Proportino of reports in a year that ended in some kind of injury
yearly_proportion=df.groupby('EventDate_year').target.mean().reset_index().rename(columns={'target':'Proportion'})

#Bin the injury rate to make it easier to read
yearly_proportion['Injury Rate']=''
yearly_proportion['Injury Rate'][(yearly_proportion['Proportion']>=0.4) & (yearly_proportion['Proportion']<0.45)] = "0+"
yearly_proportion['Injury Rate'][(yearly_proportion['Proportion']>=0.4) & (yearly_proportion['Proportion']<0.45)] = "0.4-0.45"
yearly_proportion['Injury Rate'][(yearly_proportion['Proportion']>=0.45) & (yearly_proportion['Proportion']<0.5)] = "0.45-0.5"
yearly_proportion['Injury Rate'][(yearly_proportion['Proportion']>=0.5) & (yearly_proportion['Proportion']<0.55)] = "0.5-0.55"
yearly_proportion['Injury Rate'][(yearly_proportion['Proportion']>=0.55) & (yearly_proportion['Proportion']<1)] = "0.5-0.55"
yearly_proportion['Injury Rate'][(yearly_proportion['Proportion']==1)] = "1"

#Combine the number of reports and the proportion of them that had any injury
graph1=yearly_report.merge(yearly_proportion,how='inner',on='EventDate_year')

snsplot=sns.lmplot('EventDate_year','Count', data=graph1,fit_reg=False,hue='Injury Rate')
fig=snsplot.fig
plt.title('Reports over Time',fontsize=15)
plt.xlabel('Year',fontsize=10)
plt.ylabel('Number of Reports',fontsize=10)
'''


##############################################Graph One#################################
#Sum of all injuries in a plane accident
df['Injured']=df[['TotalFatalInjuries', 'TotalMinorInjuries', 'TotalSeriousInjuries']].sum(axis=1)

#How many reports per year
yearly_report=df.groupby('EventDate_year').AccidentNumber.count().reset_index().rename(columns={'AccidentNumber':'Count'})
   
#Average number of reported fatal injuries a year
yearly_mean=df.groupby('EventDate_year').TotalFatalInjuries.mean().reset_index().rename(columns={'TotalFatalInjuries':'AvgFatal'})

#Combine the number of reports and the proportion of them that had any injury
graph1=yearly_report.merge(yearly_mean,how='inner',on='EventDate_year')

#Bin the injury rate to make it easier to read
graph1['Avg Num Fatality']=''
graph1['Avg Num Fatality'][(graph1['AvgFatal']>=0) & (yearly_proportion['AvgFatal']<0.5)] = "0-0.5"
graph1['Avg Num Fatality'][(graph1['AvgFatal']>=0.5) & (yearly_proportion['AvgFatal']<1.0)] = "0.5-1"
graph1['Avg Num Fatality'][(graph1['AvgFatal']>=1.0) & (yearly_proportion['AvgFatal']<1.5)] = "1-1.5"
graph1['Avg Num Fatality'][(graph1['AvgFatal']>=1.5) & (yearly_proportion['AvgFatal']<2.0)] = "1.5-2"
graph1['Avg Num Fatality'][(graph1['AvgFatal']>=2.0) & (yearly_proportion['AvgFatal']<2.5)] = "0.5-0.55"
graph1['Avg Num Fatality'][(graph1['AvgFatal']>=4)] = "4+"


#snsplot=sns.lmplot('EventDate_year','Count', data=graph1,fit_reg=False,hue='AvgFatal',palette="Blues")
#fig=snsplot.fig
#plt.title('Reports over Time',fontsize=15)
#plt.xlabel('Year',fontsize=10)
#plt.ylabel('Number of Reports',fontsize=10)

plt.scatter(graph1.EventDate_year, graph1.Count, alpha = .8, c = graph1.AvgFatal, cmap = 'seismic')
cbar = plt.colorbar()
plt.title('Reports over Time',fontsize=18)
plt.xlabel('Year',fontsize=15)
plt.ylabel('Number of Reports',fontsize=14)
cbar.ax.set_title('Avg Fatalities')
#plt.legend(title="Avg Fatalities")
plt.savefig('graph1.png',bbox_inches='tight')


#########################################Graph Two#########################################
#Just airplanes
planes=df[df['AircraftCategory']=='Airplane']

#Sum of all reported outcomes on a plane
planes['TotalCount']=planes[['TotalFatalInjuries','TotalSeriousInjuries', 'TotalMinorInjuries', 'TotalUninjured']].sum(axis=1)
#Sum of all injuries in a plane accident
planes['Injured']=planes[['TotalFatalInjuries', 'TotalMinorInjuries', 'TotalSeriousInjuries']].sum(axis=1)

#Sum of all injuries in a year
planes_injured_a_year=planes[['Injured','EventDate_year']].groupby('EventDate_year').sum().reset_index()
#Yearly sum of all passangers accounted for in publication
planes_total_a_year=planes[['TotalCount','EventDate_year']].groupby('EventDate_year').sum().reset_index()

#Join stats together
graph2=planes_injured_a_year.merge(planes_total_a_year,how='inner',on='EventDate_year')

#Proportion of plane passangers with an injury each year
graph2['InjuryProportion']=graph2['Injured'].div(graph2['TotalCount'])

#sns.set_style('white')
#snsplot=sns.lmplot('EventDate_year','InjuryProportion', data=graph2,fit_reg=False)
#fig=snsplot.fig
plt.scatter('EventDate_year','InjuryProportion', data=graph2)
plt.title('Proportion of Injuries from Planes over Time',fontsize=18)
plt.xlabel('Year',fontsize=15)
plt.ylabel('Proportion of Passangers Injured',fontsize=14)
plt.savefig('graph2.png',bbox_inches='tight')

