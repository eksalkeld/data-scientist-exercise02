# -*- coding: utf-8 -*-
"""
File of config values easily surfaced to an end user

Created on Wed Sep  2 19:28:37 2020

@author: eksalkeld
"""

#File and folder locations of data
REPO_DIRECTORY='C://Users//eksalkeld//Documents//GitHub//data-scientist-exercise02'
REPO_FOLDER='data'
AVIATION_EXTENSION='AviationData.xml'
AVIATION_FILE='%s//%s//%s' % (REPO_DIRECTORY,REPO_FOLDER,AVIATION_EXTENSION)
JSON_FOLDER='%s//%s' % (REPO_DIRECTORY,REPO_FOLDER)


#Define columns that need to be numeric
cols_to_numeric=['Latitude','Longitude','TotalFatalInjuries', 'TotalMinorInjuries', 'TotalSeriousInjuries','TotalUninjured','NumberOfEngines']
#Cols that have few enough levels they can be dummies (Max levels 8 except for Purpose)
cols_to_dummy=['WeatherCondition','Carrier','AmateurBuilt','Schedule','PurposeOfFlight']
#Cols that will be target encoded: 25+
cols_to_target=['Make', 'Model','AirportCode']
#Cols that will be WOE encoded
cols_to_woe=[]
#Cols that will be frequency encoded: 12-17
cols_to_freq=['AircraftCategory','BroadPhaseOfFlight','EngineType','FARDescription']

#Feature selection cutoffs
corr_threshold=0.8
chithresh=0.01
rf_top=15
miss_perc=0.8

#Logistic regression
seed=425
kfold=5
no_val_set=True
train_proportion=0.5
val_proportion=0