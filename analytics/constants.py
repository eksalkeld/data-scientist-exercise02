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
#Cols that have few enough levels they can be dummies
cols_to_dummy=[]
#Cols that will be target encoded
cols_to_target=[]
#Cols that will be WOE encoded
cols_to_woe=[]
#Cols that will be frequency encoded
cols_to_freq=[]