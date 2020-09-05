# -*- coding: utf-8 -*-
"""
Visualizations

Created on Sat Sep  5 15:12:59 2020

@author: eksalkeld
"""
import seaborn as sns
import matplotlib.pyplot as plt

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

snsplot=sns.lmplot('EventDate_year','RiskRatio', data=planes_yearly_breakdown,fit_reg=True)
fig=snsplot.fig
plt.title('Risk over Time',fontsize=15)
plt.xlabel('Year',fontsize=10)
plt.ylabel('All Injuries to Total Passangers',fontsize=10)
fig.savefig('AirplaneRiskTime2.png',bbox_inches='tight')

snsplot=sns.lmplot('EventDate_year','Injured', data=planes,fit_reg=False,hue='Carrier')
fig=snsplot.fig
plt.title('Injuries from Planes over Time',fontsize=15)
plt.xlabel('Year',fontsize=10)
plt.ylabel('Injury counts of Passangers',fontsize=10)
fig.savefig('AirplaneRiskTime2.png',bbox_inches='tight')

plane50=planes[planes['Injured']<=55]
plane50carriers=plane50[(plane50['Carrier']!='UnknownCarrier') & (plane50['Carrier']!='OtherCarrier')]

snsplot=sns.lmplot('EventDate_year','Injured', data=plane50carriers,fit_reg=False,hue='Carrier')
fig=snsplot.fig
plt.title('Injuries from Planes over Time',fontsize=15)
plt.xlabel('Year',fontsize=10)
plt.ylabel('Injury counts of Passangers',fontsize=10)

plane10=plane50carriers[plane50carriers['Injured']<=10]
snsplot=sns.lmplot('EventDate_year','Injured', data=plane10,fit_reg=False,hue='Carrier')
fig=snsplot.fig
plt.title('Injuries from Planes over Time',fontsize=15)
plt.xlabel('Year',fontsize=10)
plt.ylabel('Injury counts of Passangers',fontsize=10)
