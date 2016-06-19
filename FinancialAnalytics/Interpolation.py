
# coding: utf-8

# In[1]:

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


# In[72]:

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
    
    #print(df_null)
    #print(df_notNull)
    
    corr_index = y.corr(method = 'pearson')
    
    
    for col in df_null:
        #print(col)
        new_corr = corr_index.sort_values(by = col,axis = 0,ascending = 0)
        #print(new_corr)
        cols_highest_corr = list(new_corr)[1]
        print(cols_highest_corr)
        df_null[col].interpolate(method = 'linear',inplace = True)
        
        
    
    
        
    
    print(df_null)
    
    
    y2 = df.select_dtypes(include= [int,float])    
    
    correlations = {}
    colum=y2.columns.tolist()
    for col1,col2 in itertools.combinations(colum,2):
        correlations[col1 + '__' + col2] = pearsonr(y2.loc[:,col1],y2.loc[:,col2])
        
    result = pd.DataFrame.from_dict(correlations, orient='index')
    result.columns = ['Pear_Cor','P-value']
    
        
    
    #print(result.sort_index())


# In[73]:

autoInterpolate(df)


# In[ ]:



