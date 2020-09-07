# -*- coding: utf-8 -*-
"""
Created on Sun Sep  6 14:40:42 2020

@author: eksalkeld
"""

from constants import *

from sklearn.feature_selection import chi2
import pandas as pd
from sklearn.ensemble import RandomForestClassifier as RFC


def chi2select(df,model_cols):
    
    #Chi2 for relationship between categorical vars and target
    chi2_scores=chi2(df[model_cols],df.target)

    #Create series to hold p values with col names as index
    chi_pvals=pd.Series(chi2_scores[1],index=model_cols)

    #Find the columns that have a p value smaller than the desired cutoff (there is a relationship between var and target)
    chi_selected=chi_pvals[chi_pvals<chithresh].index
    
    #Find the unique prefixes as many of these are dummy variables
    chi_prefixes=list(set([x.split('_')[0] for x in chi_selected]))

    return chi_selected,chi_prefixes

def find_corr(df):
    
    #List to hold the names of the columns 
    corr_cols = [] 
    #Abs values of correlations
    corr_matrix = df.corr().abs()
    #Iterate through each column in corr matrix
    for i in range(len(corr_matrix.columns)):
        #This will skip the diagnol and everything above the diagnol
        for j in range(i):
            #print("i:"+str(i)+" and j:"+str(j)+" COL NAMES i:"+corr_matrix.columns[i]+" and j:"+corr_matrix.columns[j])
            #Check if the correlation between col i (listed on the corr matrix row) and col j (listed on corr matrix col) AND check if jth column already in the set 
            if (corr_matrix.iloc[i, j] >= corr_threshold) and (corr_matrix.columns[j] not in corr_cols):
                #Add the name of the column with high correlation to the list of columns
                corr_cols.append(corr_matrix.columns[i])
                
    return corr_cols

def missingcount(df):
    
    #Number of ok missing values, relative to size of dataframe
    min_missing=int(df.shape[0]*miss_perc)
    
    #List to hold names of cols that have too many missing values
    miss_cols=[]
    
    #Iterate through columns
    for i in df.columns:
        
        try:
            #For numerical columns
            if(df[i].dtype == np.float64 or np.float or df[i].dtype == np.int64 or df[i].dtype == np.int):
                #Count number of missing rows
                miss_ct=df[i].isnull().sum()
            #For boolean columns
            elif df[i].dtype==bool:
                #Set count of missing to 0
                miss_ct=0
            #For all other columns
            else:
                #Count number of missing or '' rows
                miss_ct=max(df.isnull().sum(),df[df[i]==''].shape[0])
            #If count of missing in a column is higher than acceptable append the name to the list
            if miss_ct>min_missing:
                miss_cols.append(i)
        
        #If an operation is done on a column that does not support that type of operation (=='' on a col that cannot contain '')
        except TypeError:
            pass
            
    return miss_cols


def rf_imp(X,y):
    
    #Define random forest
    rf=RFC(random_state=seed)
    
    #Fit RF to data
    rfmodel=rf.fit(X,y)
    
    #Extract importances and tie to column names
    
    
    #Find cols that are most important
    
    
    return select_vars
    
    