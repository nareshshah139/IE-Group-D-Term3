# -*- coding: utf-8 -*-
"""
Created on Wed Jun 15 21:09:06 2016

@author: rahulmehra
"""

import pandas as pd
import itertools
from scipy.stats.stats import pearsonr
import numpy as np
raw_data = {'patient': [1,np.nan, 1, 2, 2], 
        'obs': [1, 2, 3, np.nan, 2], 
        'treatment': [0.3, 1.4, 0.5, 1.2, 0.9],
        'score': ['strong', 'weak', 'normal', 'weak', 'strong'],
        'city': ['ML', 'AL', 'BL', 'CL', 'TL'],
        'date' : ['1999-05-02','1999-06-02','1999-05-03','1999-05-03','1999-05-04'] } 
df = pd.DataFrame(raw_data, columns = ['patient', 'obs', 'treatment', 'score', 'city','date'])


def autoInterpolate(y):
    
    for col in df.columns:
        if y[col].dtype == 'object':
            try:
                y[col] = pd.to_datetime(df[col])
            except ValueError:
                pass
    col_li = []
    col_nli = []
    for column_name, column in y.transpose().iterrows():
        if pd.isnull(y[column_name]).any():
            col_li.append(column_name)
        else:
            col_nli.append(column_name)
    df_null = y[col_li]
    df_notNull = y[col_nli]
    
    for col in df_null:
        df_null[col] = df_null[df_null[col] == np.nan]
        print(df_null)
    
    
    
    
    y2 = df.select_dtypes(include= [int,float])    
    
    correlations = {}
    colum=y2.columns.tolist()
    for col1,col2 in itertools.combinations(colum,2):
        correlations[col1 + '__' + col2] = pearsonr(y2.loc[:,col1],y2.loc[:,col2])
        
    result = pd.DataFrame.from_dict(correlations, orient='index')
    result.columns = ['Pear_Cor','P-value']
    
        
    
    print(result.sort_index())
         
autoInterpolate(df)            