# -*- coding: utf-8 -*-
"""
Created on Fri Oct  7 12:37:27 2022

@author: Aaron
"""

import pickle
import numpy as np
import pandas as pd
import time
import gen_Pitt_arrivals as gen_Pitt
import seq_arrival_new as seq_curb
import PAP as MOD_flex
import AP as AP
import genBids as genBids


# Variable initialization
np.random.seed(335)

tic = time.time()


#set the scenario parameters
#start and end time
start = 0
end = 1200
#number of parking spaces
c = 20
#number of vehicles arriving at the curbspace
n = 100
#number of iterations
i = 1
#schedule flexibility
phi = 5
#buffer in the optimal schedule
buffer = 0

max_hr_demand_per_space = 6
max_DVs = c * max_hr_demand_per_space * 20 #20 hours of the day from 4am to midnight


#dataframe initialization
n_index_lst_df = pd.DataFrame()
n_index_norm_lst_df = pd.DataFrame()

arrival_dfs_df = pd.DataFrame()
sum_service_df = pd.DataFrame()


#generate parking demand schedule from the Pitt dataset

#Initialize the n_index dataframe, do this upfront for the purposes of debugging
#Otherwise, the random draw is dependent on all of the other events in the loop
for c_index in range(1, c +1):

    #initialize the list to store each of the randomly drawn delivery vehicle scenarios, initialize for each new parking space
    n_index_lst = []
    n_index_norm_lst = []
    #Q data
    arrival_dfs = []
    sum_service = [] #used for graphic normalization    

    for i_index in range(1, i +1): 
        #print(i_index)
        #if i_index == 24:
            #print('stop')
        
        #next, create the vehicle arrival matrix based on a randomly determined number of deliver vehicles
        
        #what is the lower and upper bound of the number of DVs, which is dependent on the number of parking spaces
        lower_DVs = 1 #this is the lowest possible number of DVs to experience over the day, could go higher, but engineering judgement
        upper_DVs = max_hr_demand_per_space*11*c_index #we want 6 veh/hr*11hr scenario window*the number of parking spaces #added
        
        #draw a random integer between the upper and lower number of DVs to expect
        n_index = np.random.randint(lower_DVs, upper_DVs +1) #+1 becuase it is exclusive of the upper value
        n_index_lst.append(n_index)
        n_index_norm = n_index / 11 / c_index #variable available for storage
        n_index_norm_lst.append(n_index_norm)
        
        #generate a random set of vehicle arrival requests based on the number of
        #delivery vehicles in the scenario
        Q, sum_service = gen_Pitt.gen_Pitt_arrivals(n_index, end)
            
        arrival_dfs.append(Q)


    #add the populated lists of data for the current parking space to the initialized dataframes
    n_index_lst_df[c_index] = n_index_lst
    n_index_norm_lst_df[c_index] = n_index_norm_lst
    #Q data
    arrival_dfs_df[c_index] = arrival_dfs
    sum_service_df[c_index] = sum_service #used for graphic normalization


#for this stage in development, pick the last Q which corresponds to having 20 parking spaces
Q = arrival_dfs_df.iloc[0,0]

dbl_park_FCFS = []
dbl_park_Opt = []
diff_dbl_park = []

for c_index in range(2, 3):
    Q = arrival_dfs_df.iloc[0, c_index]

    #for comparison, run FCFS and just the PAP or AP which can be used against the sliding time window
    dbl_park_seq, dbl_parked_events, legal_parked_events, park_events_FCFS = seq_curb.seq_curb(c_index, Q, end)
    
    dbl_park_FCFS.append(dbl_park_seq)
    
    #Optimal model
    t_initialize = [None]
    x_initialize = [None]

    flex = phi
    n = len(Q)
    status, obj, count_b_i, end_state_t_i, end_state_x_ij, dbl_park_events, park_events \
        = MOD_flex.MOD_flex(flex, n, c_index, Q, buffer, start, end, t_initialize, x_initialize)
        
    #if the PAP times out, set the flag as true
    if status == 9:        
        #execute the AP
        x_initialize = [None] #* n_index

        bids = genBids.genBids(n, end, Q, flex)

        status, obj, dbl_park_Opt, park_demand, end_state_x_i_j, dbl_park_events, park_events = AP.AP(n_index, end, Q, c_index, bids, flex, buffer, x_initialize)
    
    
    obj = np.round(obj)
    dbl_park_Opt.append(obj)
    diff_dbl_park.append(dbl_park_seq - obj)
    
    
    
    #sliding time window model
    #define the duration for the sliding time window
    len_time_window = 240
    #establish the start time and initial number of parking spaces for the scenario
    current_time = start
    parking_spaces_available = c_index
    
    future_depart_master = pd.DataFrame()
    park_events_master = pd.DataFrame()
    
    
    #continue to step through this loop until the current time + sliding window duration is less than or equal to the end of the scenario time
    while current_time + len_time_window <= end:
        
        #determine the start and end of the time window and subset Q accordingly
        window_start = current_time
        window_end = current_time + len_time_window
        Q_window = Q[(Q['a_i'] >= window_start) & (Q['a_i'] < window_end)]
        
        #optimize with PAP or AP the event occuring in the current time window
        t_initialize = [None] #added
        x_initialize = [None] #added
        n = len(Q_window)
        c = parking_spaces_available
        
        #if n  =  1 and there is a parking space available, then assign n
        #if n  = 1 but there isn't a parking space, dbl park the vehicle
        #if c = 0, dbl park and vehicles
        #if n = 0, it doesn't really matter, no care to consider over this timeframe?  But need to increase parking spaces available potentially
        
        
        if (n != 0) & (n != 1) & (c != 0):
            status, obj, count_b_i, end_state_t_i, end_state_x_ij, dbl_park_events, park_events \
                = MOD_flex.MOD_flex(flex, n, c, Q_window, buffer, window_start, window_end, t_initialize, x_initialize)
            
            #store the current event schedule and combine with the events from the previous time step
            park_events_master = pd.concat([park_events_master, park_events], axis = 0, ignore_index = True)
                
            #identify vehicle arrivals during this time window that will depart in a future time window
            future_depart = park_events[(park_events['Park Type'] == 'Legal Park') &
                                         (park_events['d_i'] >= window_end)
                                         ]
            #add these future departure events to a master departure dataframe
            future_depart_master = pd.concat([future_depart_master, future_depart], axis = 0, ignore_index = True)
            
            
            #how many legal parking arrival occur during this time step's optimal schedule?
            num_arrivals = len(park_events[park_events['Park Type'] == 'Legal Park'])
            #how many legal parking departures occur during this time step's optimal schedule?
            num_depart = len(park_events[(park_events['Park Type'] == 'Legal Park') &
                                         (park_events['d_i'] < window_end)
                                         ])
            #capture any departure events that were stored in the master departure dataframe,
            #only consider the departure events that are between the current window start and end
            num_depart_master = len(future_depart_master[(future_depart_master['d_i'] < window_end) &
                                                         (future_depart_master['a_i'] >= window_start)
                                                         ])       
            #calculate and apply the change in occupancy during this time period
            delta_occupancy = -num_arrivals + num_depart + num_depart_master
            parking_spaces_available += delta_occupancy
            
            
            
            #step the current time forward to the end of the current window
            current_time += len_time_window












