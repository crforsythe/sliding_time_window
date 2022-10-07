# -*- coding: utf-8 -*-
"""
Created on Sat Sep 10 13:40:21 2022

@author: Aaron
"""


def gen_Pitt_arrivals(n_index, end):
    
    import numpy as np
    import pandas as pd
    import random
    import pickle
    from datetime import datetime

    #n_index = 5
    #end = 1200 #from 4am through Midnight

    
    with open('Pitt_truck_data_final.pkl', 'rb') as file:
            raw_df, \
            truck_data_final \
                = pickle.load(file)
                
    
    #generate random integers between 0 and the length of truck_data_final that can
    #be used as row indexes
    random_idx = np.random.randint(0, len(truck_data_final)-1, n_index)
    
    
    a_i = []
    s_i = []
        
    #random_idx = [0, 1]
    
    for item in random_idx:
        
        #this flag is active until the random start_window + service duration is
        #verified to not be longer than the end of the scenario.  If the sum is
        #larger than the end of the scenario, then the script will redraw new 
        #values and go through the verification process again
        flag = True
        
        while flag == True:
            
            draw_hr = truck_data_final['arrive_time_formatted'].iloc[item].hour
            draw_hr_convt = (draw_hr - 4) * 60 #convert to minutes from the start of the scenario at 4am
            draw_min = truck_data_final['arrive_time_formatted'].iloc[item].minute
            arrival_time = draw_hr_convt + draw_min
            
            draw_si = int(truck_data_final['duration'].iloc[item] / 60) #convert to minutes and to an integer value
            
            if arrival_time + draw_si > end:
                item = random.randint(0, len(truck_data_final)) #not a good draw, arrival + service time exceed the end of the scenario, e.g. midnight, redraw a row index
            else:
                flag = False #good sample, change the flag so that we can exit the while loop and store the current truck data
            
        a_i.append(arrival_time)
        s_i.append(draw_si)
        
        
    #create a dictionary from lists of data that will be turned into a dataframe
    requests = {'a_i': a_i, 'b_i': a_i, 's_i': s_i, 't_i': a_i}
    
    #create index label for Q
    Trucks = ['Truck_' + str(i+1) 
          for i in range(0, len(requests['a_i']))]
    
    Q = pd.DataFrame(requests, index = Trucks)
    
    Q['d_i'] = Q['a_i'] + Q['s_i']
    
    #return the sum of the service durations for the instance
    sum_s = sum(s_i)
                
    
    return Q, sum_s