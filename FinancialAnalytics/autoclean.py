# -*- coding: utf-8 -*-
"""
Created on Wed Jun 15 11:39:04 2016

@author: rahulmehra
"""




import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np
raw_data = {'patient': [500,1, 1,-2, 2,3,4,5,6,7,None,9,6,3,4,5,6,7,8,9], 
        'obs': [1, 2, 3, 1, 2,3,4,5,6,7,8,9,6,3,4,5,6,7,8,9], 
        'treatment': [0, 1, 0, 1, 0,0, 1, 0, 1, 0,0, 1, 0, 1, 0,0, 1, 0, 1, 0],
        'score': ['strong', 'weak', 'normal', 'weak', 'strong','strong', 'weak', 'normal', 'weak', 'strong','strong', 'weak', 'normal', 'weak', 'strong','strong', 'weak', 'normal', 'weak', 'strong'],
        'city': ['ML', 'AL', 'BL', 'CL', 'TL','ML', 'AL', 'BL', 'CL', 'TL','ML', 'AL', 'BL', 'CL', 'TL','ML', 'AL', 'BL', 'CL', 'TL']} 
input_dataframe = pd.DataFrame(raw_data, columns = ['patient', 'obs', 'treatment', 'score', 'city'])

def autoclean(x):
    #if copy:
        #input_dataframe = input_dataframe.copy()
        
    #if drop_nans:
        #input_dataframe.dropna(inplace=True)
    for column in x.columns:
        print(x[column])
        # Replace NaNs with the median or mode of the column depending on the column type
       
        
        if x[column].dtypes == 'int64' or x[column].dtypes == 'float64':
            print("I'm here")
            
            mean = x[column].mean()
            std = 1.5*x[column].std()
            x[column] = x[column].apply(lambda y: x[column].median() if(abs(y - mean >std)) else y)
            
            n_rows = len(input_dataframe.index)
            negative_perc = np.sum((x[column] < 0))/n_rows
            x[column] = x[column].apply(lambda y: -(y) if (y<0 and negative_perc >= 0.05) else y)
            
            x[column]
        # Encode all strings with numerical equivalents
        if str(x[column].values.dtype) == 'object':
            column_encoder = LabelEncoder().fit(x[column].values)

            x[column] = column_encoder.transform(x[column].values)
        print(x[column].dtype)
        
        
    
    
    return x

autoclean(input_dataframe)
print(input_dataframe)    
 #apply(func[, convert_dtype, args])   
                  
                  