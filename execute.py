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
c = 10
#number of vehicles arriving at the curbspace
n = 100
#number of iterations
i = 10
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


#prepare for the execute loop

dbl_park_FCFS_df = pd.DataFrame()
dbl_park_Opt_df = pd.DataFrame()
dbl_park_sliding_df = pd.DataFrame()
redux_FCFS_Opt_df = pd.DataFrame()
redux_FCFS_sliding_df = pd.DataFrame()
redux_Opt_sliding_df = pd.DataFrame()


for c_index in range(2, 3):
    
    dbl_park_FCFS = []
    dbl_park_Opt = []
    dbl_park_sliding = []
    redux_FCFS_Opt = []
    redux_FCFS_sliding = []
    redux_Opt_sliding = []
    
    for i_index in range(0, 5):
        
        print('c = ', c_index, 'i = ', i_index)
        
        #pull the correct vehicle request matrix
        Q = arrival_dfs_df.iloc[i_index, c_index]
        
        #for comparison, run FCFS and just the PAP or AP which can be used against the sliding time window
        dbl_park_seq, dbl_parked_events, legal_parked_events, park_events_FCFS = seq_curb.seq_curb(c_index, Q, end)
        
        dbl_park_FCFS.append(dbl_park_seq)
        
        
        #Optimal model
        t_initialize = [None]
        x_initialize = [None]
        
        flex = phi
        n = len(Q)
        status, obj, count_b_i, end_state_t_i, end_state_x_ij, dbl_park_events, park_events \
            = MOD_flex.MOD_flex(flex, n, c_index, Q, buffer, start, end, end, t_initialize, x_initialize)
            
        #if the PAP times out, set the flag as true
        if status == 9:        
            #execute the AP
            x_initialize = [None] #* n_index
        
            bids = genBids.genBids(n, end, Q, flex)
        
            status, obj, dbl_park_Opt_AP, park_demand, end_state_x_i_j, dbl_park_events, park_events = AP.AP(n, end, Q, c_index, bids, flex, buffer, x_initialize)
        
        
        obj = np.round(obj)
        dbl_park_Opt.append(obj)

        
        
        
        #sliding time window model
        #define the duration for the sliding time window
        len_time_window = 30
        #establish the start time and initial number of parking spaces for the scenario
        current_time = start
        parking_spaces_available = c_index
        
        future_depart_master = pd.DataFrame(columns = ['Truck', 'a_i', 's_i', 'd_i', 'Park Type'])
        park_events_master = pd.DataFrame()
        
        
        #continue to step through this loop until the current time + sliding window duration is less than or equal to the end of the scenario time
        while current_time + len_time_window <= end:
            
            #determine the start and end of the time window and subset Q accordingly
            window_start = current_time
            window_end = current_time + len_time_window
            Q_window = Q[(Q['a_i'] >= window_start) & (Q['a_i'] < window_end)]
            print(window_start, ' ', window_end)
            #if window_start == 635:
                #print('stop')
            
            #optimize with PAP or AP the event occuring in the current time window
            t_initialize = [None] #added
            x_initialize = [None] #added
            n = len(Q_window)
            c = parking_spaces_available
            
            #if n  =  1 and there is a parking space available, then assign n
            #if n  = 1 but there isn't a parking space, dbl park the vehicle
            #if c = 0, dbl park and vehicles
            #if n = 0, it doesn't really matter, no care to consider over this timeframe?  But need to increase parking spaces available potentially
            
            if c == 0: #no parking spaces are available, so double park all of the vehicles in Q_window
                #reformat the Q_window set of vehicle during this time window so that it can be appended to park_events_master
                dbl_park_vehicles = Q_window.drop(columns = ['b_i', 't_i'])
                dbl_park_vehicles['Park Type'] = 'Dbl Park'
                dbl_park_vehicles.rename(columns = {'Trucks':'Truck'}, inplace = True)
                
                #store the current event schedule and combine with the events from the previous time step
                park_events_master = pd.concat([park_events_master, dbl_park_vehicles], axis = 0, ignore_index = True)
                
                #still check to see if there were any future departures that occur during this time window
                num_depart_master = len(future_depart_master[(future_depart_master['d_i'] <= window_end) &
                                                             (future_depart_master['d_i'] > window_start)
                                                             ])   
                
                #calculate and apply the change in occupancy during this time period
                delta_occupancy = num_depart_master
                parking_spaces_available += delta_occupancy
                
                
                #step the current time forward to the end of the current window
                current_time += len_time_window
                
                
            elif n == 0: #no parking requests during this time window
                #still check to see if there are any future departures
                num_depart_master = len(future_depart_master[(future_depart_master['d_i'] <= window_end) &
                                                             (future_depart_master['d_i'] > window_start)
                                                             ])   
                
                #calculate and apply the change in occupancy during this time period
                delta_occupancy = num_depart_master
                parking_spaces_available += delta_occupancy
                
                
                #step the current time forward to the end of the current window
                current_time += len_time_window
                
            
            elif (n == 1) & (parking_spaces_available > 0): #trivial problem, no optimization needed, just assign the single truck to a parking space
                
                legal_park_vehicle = Q_window.drop(columns = ['b_i', 't_i'])
                legal_park_vehicle['Park Type'] = 'Legal Park'
                legal_park_vehicle.rename(columns = {'Trucks':'Truck'}, inplace = True)
                
                #store the current event schedule and combine with the events from the previous time step
                park_events_master = pd.concat([park_events_master, legal_park_vehicle], axis = 0, ignore_index = True)
            
                #identify vehicle arrivals during this time window that will depart in a future time window
                future_depart = Q_window[(Q_window['d_i'] > window_end)]
                future_depart = future_depart.drop(columns = ['b_i', 't_i'])
                future_depart['Park Type'] = 'Legal Park'
                future_depart.rename(columns = {'Trucks': 'Truck'}, inplace = True)
                
                #add these future departure events to a master departure dataframe
                future_depart_master = pd.concat([future_depart_master, future_depart], axis = 0, ignore_index = True)
                
                #how many legal parking arrival occur during this time step's optimal schedule?
                num_arrivals = len(legal_park_vehicle[legal_park_vehicle['Park Type'] == 'Legal Park'])
                #how many legal parking departures occur during this time step's optimal schedule?
                num_depart = len(legal_park_vehicle[(legal_park_vehicle['Park Type'] == 'Legal Park') &
                                             (legal_park_vehicle['d_i'] <= window_end)
                                             ])
                #capture any departure events that were stored in the master departure dataframe,
                #only consider the departure events that are between the current window start and end
                num_depart_master = len(future_depart_master[(future_depart_master['d_i'] <= window_end) &
                                                             (future_depart_master['d_i'] > window_start)
                                                             ])   
                
                #calculate and apply the change in occupancy during this time period
                delta_occupancy = -num_arrivals + num_depart + num_depart_master
                parking_spaces_available += delta_occupancy
                
                
                #step the current time forward to the end of the current window
                current_time += len_time_window
                
                
            elif (n != 0) & (n != 1) & (c != 0):
                status, obj, count_b_i, end_state_t_i, end_state_x_ij, dbl_park_events, park_events \
                    = MOD_flex.MOD_flex(flex, n, c, Q_window, buffer, window_start, window_end, end, t_initialize, x_initialize)
                    
                #if the PAP times out, set the flag as true
                #before this can be added, the AP needs to be updated to receive a window_start parameter
                # if status == 9:        
                #     #execute the AP
                #     x_initialize = [None] #* n_index
        
                #     bids = genBids.genBids(n, end, Q_window, flex)
        
                #     status, obj, dbl_park_Opt, park_demand, end_state_x_i_j, dbl_park_events, park_events = AP.AP(n, window_end, Q_window, c, bids, flex, buffer, x_initialize)
                
                
                #store the current event schedule and combine with the events from the previous time step
                park_events_master = pd.concat([park_events_master, park_events], axis = 0, ignore_index = True)
                    
                #identify vehicle arrivals during this time window that will depart in a future time window
                future_depart = park_events[(park_events['Park Type'] == 'Legal Park') &
                                             (park_events['d_i'] > window_end) #> and not >= window end, b/c departures should be considered first, e.g. depart at 720 counts first before an arrival in the next window which starts at 720
                                             ]
                
                #add these future departure events to a master departure dataframe
                future_depart_master = pd.concat([future_depart_master, future_depart], axis = 0, ignore_index = True)
                
                
                #how many legal parking arrival occur during this time step's optimal schedule?
                num_arrivals = len(park_events[park_events['Park Type'] == 'Legal Park'])
                #how many legal parking departures occur during this time step's optimal schedule?
                num_depart = len(park_events[(park_events['Park Type'] == 'Legal Park') &
                                             (park_events['d_i'] <= window_end)
                                             ])
                #capture any departure events that were stored in the master departure dataframe,
                #only consider the departure events that are between the current window start and end
                num_depart_master = len(future_depart_master[(future_depart_master['d_i'] <= window_end) &
                                                             (future_depart_master['d_i'] > window_start)
                                                             ])       
                #calculate and apply the change in occupancy during this time period
                delta_occupancy = -num_arrivals + num_depart + num_depart_master
                parking_spaces_available += delta_occupancy
                
                
                
                #step the current time forward to the end of the current window
                current_time += len_time_window
                
                
        #record the total amount of double parking
        dbl_park_sliding = np.sum(park_events_master[park_events_master['Park Type'] == 'Dbl Park']['s_i'])
        
        
        #store the diff/redux data from the current iteration in a list to later be appended to the larger dataframe
        redux_FCFS_Opt.append(dbl_park_seq - obj)
        redux_FCFS_sliding.append(dbl_park_seq - dbl_park_sliding)
        redux_Opt_sliding.append(obj - dbl_park_sliding)
        
        
    #add all of the data from the current set of iterations to the column which corresponds to the current parking space loop
    dbl_park_FCFS_df[c_index] = dbl_park_FCFS
    dbl_park_Opt_df[c_index] = dbl_park_Opt
    dbl_park_sliding_df[c_index] = dbl_park_sliding
    redux_FCFS_Opt_df[c_index] = redux_FCFS_Opt
    redux_FCFS_sliding_df[c_index] = redux_FCFS_sliding
    redux_Opt_sliding_df[c_index] = redux_Opt_sliding
    




toc = time.time()

runtime = toc-tic
print('runtime: ' + str(runtime))


# import pickle
# with open('sliding_window_30.pkl', 'wb') as file: 
#     pickle.dump(
#         [redux_FCFS_Opt,
#           redux_FCFS_Opt_df,
#           redux_FCFS_sliding,
#           redux_FCFS_sliding_df,
#           redux_Opt_sliding,
#           redux_Opt_sliding_df,
#           arrival_dfs_df, 
#           sum_service_df, 
#           runtime, 
#           buffer,
#           end,
#           i,
#           c,
#           phi          
#           ],
#             file)





