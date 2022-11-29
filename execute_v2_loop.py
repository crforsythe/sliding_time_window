# -*- coding: utf-8 -*-
"""
Created on Thu Oct 27 12:39:15 2022

@author: Aaron
"""

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
import PAP as MOD_flex
import seq_arrival_new as seq_curb

tic = time.time()

#set the scenario parameters
#start and end time
start = 0
end_scenario = 60
#number of parking spaces
c = 2
#number of vehicles arriving at the curbspace
n = 100
#number of iterations
i = 10
#buffer in the optimal schedule
buffer = 0
#generic schedule flexibility
phi = 5


# for testing, setup the truth vehicle request matrix
#vehicle_label = ['Veh1', 'Veh2', 'Veh3', 'Veh4', 'Veh5', 'Veh6', 'Veh7', 'Veh8', 'Veh9', 'Veh10']
vehicle_label = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
recieved = [1, 2, 3, 7, 8, 8, 9, 10, 11, 13]
a_i_OG = [3, 7, 9, 12, 9, 22, 20, 12, 16, 13]
s_i = [5, 13, 10, 3, 4, 2, 7, 5, 4, 3]
d_i_OG = [8, 20, 19, 15, 13, 24, 27, 17, 20, 16]
phi = [5, 5, 5, 5, 5, 5, 5, 5, 5, 5]

req_truth = pd.DataFrame(list(zip(vehicle_label, recieved, a_i_OG, s_i, d_i_OG, phi)),
                         columns = ['Vehicle','Received', 'a_i_OG', 's_i', 'd_i_OG', 'phi'])

#length of time to collect parking requests
#zeta = 1
zeta_lst = [.25, .5, 1]
#length of time to schedule requests in the future
#tau = 30
tau_lst = [1, 5, 10, 20, 30, 40, 50, 60]


sliding_legal_park_df = pd.DataFrame(index = tau_lst)
sliding_dbl_park_df = pd.DataFrame(index = tau_lst)


for zeta in zeta_lst:
    
    sliding_legal_park = []
    sliding_dbl_park = []
    
    for tau in tau_lst:
        
        #initialize the necessary tracking matricies
        req_master = pd.DataFrame(columns = ['Vehicle','Received', 'a_i_OG', 's_i', 'd_i_OG', 'phi',
                                             'Assigned', 'a_i', 'd_i', 't_i'])
        
        opt_counter = 0 #counter to track the number of times we run the optimization algorithm
        current_time = start
        
        
        while current_time < end_scenario:
            
            current_time = current_time + zeta
            print('\nCurrent_time = ', + current_time)
            print('End of window = ', + current_time + tau)
            
            #identify any requests that have been received between the the current time - zeta and the current time
            req_incoming = req_truth.loc[np.where((req_truth['Received'] >= current_time - zeta) &
                                                  (req_truth['Received'] < current_time)
                                                  )]
            req_incoming.reset_index(inplace = True, drop = True)
            #store the new incoming requests in the master requests list
            req_master = pd.concat([req_master, req_incoming])
            req_master.reset_index(inplace = True, drop = True)
            
            #identify the incoming requests within the optimization time window under consideration, tau.  Can include request with
            #a_i_OG before current_time as long as phi allows the request to be scheduled within current_time + tau.
            #The 'isna' check allows for incoming requests that were originally outside of current_time + tau to come back later in another
            #future time window.  They aren't "new" incoming per se, but they are old incoming requests that have not yet been processed
            #or considered in an optimal schedule.
            req_incoming_tau = req_master.loc[np.where(
                                                            ((req_master['Assigned'].isna() == True) & #this request is nan, not 'No' or 'Yes' to being assigned, e.g. this request has never been input to the optimization framework                           
                                                             (req_master['a_i_OG'] >= current_time) & #request is greater than current time
                                                             (req_master['a_i_OG'] + req_master['phi'] < current_time + tau)) #but request arrival + flex is less than current time + tau
                                                         |                                                                  #OR
                                                             ((req_master['Assigned'].isna() == True) & #this request is nan, not 'No' or 'Yes' to being assigned, e.g. this request has never been input to the optimization framework       
                                                              (req_master['a_i_OG'] + req_master['phi'] >= current_time) & #requested arrival + flex is greater than current time
                                                              (req_master['a_i_OG'] + req_master['phi'] < current_time + tau)) #but requested arrival + flex is less than current time + tau
                                                         )]
            #req_incoming_tau['Assigned'] = 'No'
            
            # #identify previously received, but not yet processed requests, e.g. a request sent in well in advance of a_i_OG
            # req_old_nan_tau = req_master.loc[np.where(
            #                                             ((req_master['Assigned'].isna() == True) &
            #                                             (req_master['a_i_OG'] >= current_time) & #request is greater than current time
            #                                             (req_master['a_i_OG'] + req_master['phi'] < current_time + tau)) #but request arrival + flex is less than current time + tau
            #                                             |
            #                                             ((req_master['Assigned'].isna() == True) &
            #                                             (req_master['a_i_OG'] + req_master['phi'] >= current_time) & #requested arrival + flex is greater than current time
            #                                             (req_master['a_i_OG'] + req_master['phi'] < current_time + tau)) #but requested arrival + flex is less than current time + tau
            #                                             )]
            
            #identify any old and not assigned vehicles which might still be relevant in the current time window
            req_old_no_tau = req_master.loc[np.where(
                                                    (req_master['a_i_OG'] + req_master['phi'] >= current_time) & #vehicle request parking + flex is greater than current time
                                                    (req_master['a_i_OG'] + req_master['phi'] < current_time + tau) & #but the vehicle request + flex is less than current time + tau
                                                    (req_master['Assigned'] == 'No') #the vehicle was not previously assigned a parking space
                                                    )]
            req_old_no_tau.reset_index(inplace = True, drop = True)
            
            #identify and update vehicles which were legally assigned parking previously and are still present in the current tua window
            req_old_yes_tau = req_master.loc[np.where(
                                                        (req_master['Assigned'] == 'Yes') & #vehicle was assigned parking
                                                        (req_master['a_i'] < current_time) & #vehicle started parking prior to current time
                                                        (req_master['d_i'] > current_time) #vehicle is departing after the current time
                                                        )]
            req_old_yes_tau.reset_index(inplace = True, drop = True)
            
            #need to edit parameters required to include these vehicle into Q, update relative to current_time
            req_old_yes_tau['a_i_OG'] = current_time
            req_old_yes_tau['s_i'] = req_old_yes_tau['d_i'] - current_time
            req_old_yes_tau['d_i_OG'] = req_old_yes_tau['d_i']
            req_old_yes_tau['phi'] = 0
            
            #identify vehicles which have not started parking yet, but are legally assign to park in the current tau window
            req_old_yes_future_tau = req_master.loc[np.where(
                                                        (req_master['Assigned'] == 'Yes') & #vehicle was assigned parking
                                                        (req_master['a_i'] >= current_time) & #vehicle is starting parking after current time
                                                        (req_master['a_i'] < current_time + tau) #vehicle should be starting before current time + tau, starting within the current window
                                                        )]
            req_old_yes_future_tau.reset_index(inplace = True, drop = True)
            
            #need to edit parameters required to include these vehicle into Q, update relative to current_time
            req_old_yes_future_tau['a_i_OG'] = req_old_yes_future_tau['a_i']
            req_old_yes_future_tau['d_i_OG'] = req_old_yes_future_tau['d_i']
            req_old_yes_future_tau['phi'] = 0
            
            #combine these previous events that are held over in a new matrix which will be input for the unique requirements to the PAP
            
            #combine and convert the set of requests between current time and += tau to go into the PAP
            Q = pd.DataFrame(columns = ['Vehicle','a_i', 'b_i', 's_i', 't_i', 'd_i', 'phi', 'Prev Assigned'])
            Q['Vehicle'] = pd.concat([req_incoming_tau['Vehicle'], req_old_no_tau['Vehicle'], req_old_yes_tau['Vehicle'], req_old_yes_future_tau['Vehicle']])
            Q['a_i'] = pd.concat([req_incoming_tau['a_i_OG'], req_old_no_tau['a_i_OG'], req_old_yes_tau['a_i_OG'], req_old_yes_future_tau['a_i_OG']])   
            Q['b_i'] = pd.concat([req_incoming_tau['a_i_OG'], req_old_no_tau['a_i_OG'], req_old_yes_tau['a_i_OG'], req_old_yes_future_tau['a_i_OG']])   
            Q['s_i'] = pd.concat([req_incoming_tau['s_i'], req_old_no_tau['s_i'], req_old_yes_tau['s_i'], req_old_yes_future_tau['s_i']])  
            Q['t_i'] = pd.concat([req_incoming_tau['a_i_OG'], req_old_no_tau['a_i_OG'], req_old_yes_tau['a_i_OG'], req_old_yes_future_tau['a_i_OG']])  
            Q['d_i'] = pd.concat([req_incoming_tau['d_i_OG'], req_old_no_tau['d_i_OG'], req_old_yes_tau['d_i_OG'], req_old_yes_future_tau['d_i_OG']])   
            Q['phi'] = pd.concat([req_incoming_tau['phi'], req_old_no_tau['phi'], req_old_yes_tau['phi'], req_old_yes_future_tau['phi']])
            Q['Prev Assigned'] = pd.concat([req_incoming_tau['Assigned'], req_old_no_tau['Assigned'], req_old_yes_tau['Assigned'], req_old_yes_future_tau['Assigned']])
            Q.sort_values(by = ['Vehicle'], inplace = True)
            Q.reset_index(inplace = True, drop = True)
            
            n_tau = len(Q)
            t_initialize = None
            x_initialize = None
            
            #run the PAP
            if Q.empty == False:
                status, obj, count_b_i, end_state_t_i, end_state_x_ij, dbl_park_events, park_events \
                    = MOD_flex.MOD_flex(n_tau, c, Q, buffer, current_time, current_time+tau, end_scenario, t_initialize, x_initialize)
                opt_counter += 1
            
                #step through the legally parked vehicles and record the information back to the master requests dataframe
                #likely do not need to step through the dbl parked vehicle because they will be continually reshuffled and possibly added in the future
                for item in range(0, len(park_events)):
                    #what is the current vehicle
                    current_veh = park_events.iloc[item]['Vehicle']
                    #what is the index of the current vehicle in the master requests dataframe?
                    idx = req_master[req_master['Vehicle'] == current_veh].index.values[0]
                    #record the values from the optimal solution in the master request dataframe
                    #however, first check to see if this vehicle has been assigned a start time or not (one option: np.isnan(req_master.iloc[2]['a_i']))
                    if req_master.iloc[idx]['Assigned'] != 'Yes':
                        if park_events.iloc[item]['Park Type'] == 'Legal Park':
                            req_master.iloc[idx]['Assigned'] = 'Yes'
                            req_master.iloc[idx]['a_i'] = park_events.iloc[item]['a_i']
                            req_master.iloc[idx]['d_i'] = park_events.iloc[item]['d_i']
                        elif park_events.iloc[item]['Park Type'] == 'No Park':
                            req_master.iloc[idx]['Assigned'] = 'No'
        
        
        #capture the data
        sliding_legal_park.append(np.sum(req_master[req_master['Assigned'] == 'Yes']['s_i']))
        sliding_dbl_park.append(np.sum(req_master[req_master['Assigned'] == 'No']['s_i']))
        
    sliding_legal_park_df[zeta] = sliding_legal_park
    sliding_dbl_park_df[zeta] = sliding_dbl_park
        
        
        
#FCFS
Q_FCFS = pd.DataFrame(columns = ['Vehicle','a_i', 'b_i', 's_i', 't_i', 'd_i', 'phi', 'Prev Assigned'])
Q_FCFS['Vehicle'] = req_master['Vehicle']
Q_FCFS['a_i'] = req_master['a_i_OG']
Q_FCFS['b_i'] = req_master['a_i_OG']
Q_FCFS['s_i'] = req_master['s_i']
Q_FCFS['t_i'] = req_master['a_i_OG']
Q_FCFS['d_i'] = req_master['d_i_OG']
Q_FCFS['phi'] = phi
Q_FCFS['Prev Assigned'] = 'nan'

dbl_park_seq, dbl_parked_events, legal_parked_events, park_events_FCFS = seq_curb.seq_curb(c, Q_FCFS, end_scenario)


#Optimal PAP
n = len(Q_FCFS)
status, obj, count_b_i, end_state_t_i, end_state_x_ij, dbl_park_events, park_events \
    = MOD_flex.MOD_flex(n, c, Q_FCFS, buffer, start, end_scenario, end_scenario, t_initialize, x_initialize)
        
        
# print('\nSliding Time Window Metrics:')
# print('zeta = ' + str(zeta) + ', tau = ' + str(tau))
# print('Number of Optimizations = ' + str(opt_counter))
# print('Total s_i = ' + str(np.sum(req_master['s_i'])))
# print('Legal Park = ' + str(np.sum(req_master[req_master['Assigned'] == 'Yes']['s_i'])))
# print('Not Assigned = ' + str(np.sum(req_master[req_master['Assigned'] == 'No']['s_i'])))

print('\nFCFS Metrics:')
print('Legal Park = ' + str(np.sum(park_events_FCFS[park_events_FCFS['Park Type'] == 'Legal Park']['s_i'])))
print('Not Assigned = ' + str(np.sum(park_events_FCFS[park_events_FCFS['Park Type'] == 'Dbl Park']['s_i'])))

print('\nOptimal Metrics:')
print('Legal Park = ' + str(np.sum(park_events[park_events['Park Type'] == 'Legal Park']['s_i'])))
print('Not Assigned = ' + str(np.sum(park_events[park_events['Park Type'] == 'Dbl Park']['s_i'])))

        

toc = time.time()

runtime = toc-tic
print('\nruntime: ' + str(runtime))




#plot    
data_df = pd.DataFrame(columns = ['Zeta', 'Tau', 'Legal Park', 'Dbl Park'])
row = 0

for j in range(0, sliding_legal_park_df.shape[1]):
    for i in range(0, sliding_legal_park_df.shape[0]):
        #which zeta and tau are we on?
        z = zeta_lst[j]
        t = tau_lst[i]
        
        data_df.loc[row, 'Zeta'] = z
        data_df.loc[row, 'Tau'] = t
        data_df.loc[row, 'Legal Park'] = sliding_legal_park_df.iloc[i,j]
        data_df.loc[row, 'Dbl Park'] = sliding_dbl_park_df.iloc[i,j]
        
        row += 1
        

FCFS_lower_bound = np.sum(park_events_FCFS[park_events_FCFS['Park Type'] == 'Legal Park']['s_i'])      
Opt_upper_bound = np.sum(park_events[park_events['Park Type'] == 'Legal Park']['s_i'])
        

plt.figure()
sns.lineplot(data = data_df, x = 'Tau', y = 'Legal Park', hue = 'Zeta', style = 'Zeta', markers = True, palette = "tab10")
plt.ylim([0, park_events['s_i'].sum()*1.1])
plt.axhline(y = FCFS_lower_bound, linewidth=2, color='r', linestyle = '--', label = 'FCFS')
plt.axhline(y = Opt_upper_bound, linewidth=2, color='g', linestyle = '--', label = 'Opt')
plt.legend(title = 'Zeta', loc = 'lower right', prop={'size': 8})
plt.xlabel('Tau, how many minutes into the future do we want to consider parking requests')
plt.ylabel('Minutes of legal parking scheduled')
plt.title('Minutes of Legal Parking comparison between sliding time window, FCFS, and Opt \n (Number of Parking Spaces = ' + str(c) + ')')













