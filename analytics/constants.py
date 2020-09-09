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

###Feature prep
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

####Feature selection cutoffs
#Do not keep cols with corr above this val
corr_threshold=0.8
#Features are important if the p value of their chi sq with target is below
chithresh=0.01
#Take the top x variable rated important by a random forest
rf_top=15
#If a col has more than this percent of values missing, discard
miss_perc=0.8

###Logistic regression
#Seed for reproducability
seed=425
#Folds for cv
kfold=5
#True if just train/test split, not a val too
no_val_set=True
#Proportion of data to go into train
train_proportion=0.5
#If val set created, proportion to go into val
val_proportion=0
#Penalty for LR - if l1 added solver may need to be adjusted depending on package version
penalty = ['l2']
#Whether to account for unbalanced classes, additional value is None
classweight=['balanced']
#Regularization
c = [1]
    