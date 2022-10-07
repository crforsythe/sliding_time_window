# -*- coding: utf-8 -*-
"""
Created on Wed Oct 27 20:03:08 2021

@author: Burns
"""

def genBids(num_trucks, end, Q, flex):
  
    #num_trucks = 3
    #end = 7
      
    import numpy as np
    import pandas as pd
    
    #bids and AP model require that space be descritized into time steps
    Q['a_i'] = Q['a_i'].astype(int)
    Q['b_i'] = Q['b_i'].astype(int)
    Q['s_i'] = Q['s_i'].astype(int)
    Q['t_i'] = Q['t_i'].astype(int)
    Q['d_i'] = Q['d_i'].astype(int)
    
    # Q['a_i'] = Q.loc[:, 'a_i'].astype(int)
    # Q.loc[:, 'b_i'].astype(int)
    # Q.loc[:, 's_i'].astype(int)
    # Q.loc[:, 't_i'].astype(int)
    # Q.loc[:, 'd_i'].astype(int)
    
    # Q.at[:, 'a_i'].astype(int, copy = False)
    
    
    bids = pd.DataFrame()
    
    for i in range(1, num_trucks +1): #iterate over the number of trucks in the scenario
        a = np.zeros(end +1) #create the bid vector for truck i
        a_start = Q['a_i'][i-1] #find the time/place in the vector where a_i is located
        a[a_start] = Q['s_i'][i-1] #set a_i = total service duration s_i
        
        if flex != 0: #expand the range of parking request by flex, valued at the total service duration
            for phi in range(1, flex +1):
                if (a_start +phi > end):
                    break
                else:
                    a[a_start +phi] = Q['s_i'][i-1]
                    
            for phi in range(1, flex +1):
                if a_start -phi < 0:
                    break
                else:
                    a[a_start -phi] = Q['s_i'][i-1]
        
        bids[i-1] = a #store the newly created a in the larger bids dataframe
    
  
    
    return bids #, s_i