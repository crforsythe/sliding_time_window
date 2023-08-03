# -*- coding: utf-8 -*-
"""
Created on Wed Nov 17 10:35:35 2021

@author: Burns
"""



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math as math



# num_spaces = 1
# a_i = [5, 10, 25]
# s_i = [25, 5, 5]

# #create a dictionary from lists of data that will be turned into a dataframe
# requests = {'a_i': a_i, 'b_i': a_i, 's_i': s_i, 't_i': a_i}

# #create index label for Q
# Trucks = ['Truck_' + str(i+1) 
#       for i in range(0, len(requests['a_i']))]

# Q = pd.DataFrame(requests, index = Trucks)

# Q['d_i'] = Q['a_i'] + Q['s_i']


def seq_curb(num_spaces, Q, end):

    # try:
        
        # if Q['Trucks'].all() == None:
        #     #adaptations to the received Q matrix to aid truck identification later in the function
        Q.insert(0, 'Trucks', Q.index)
        Q.reset_index(level = 0, drop = True, inplace = True)
        
        
        #gather the arrivals and departure, combine them into a list and sort
        #from earliest to latest
        Arrivals = pd.DataFrame(Q['Trucks'])
        Arrivals['t_i'] = Q['a_i']
        Arrivals['Event'] = "Arrival"
        
        Depart = pd.DataFrame(Q["Trucks"])
        Depart['t_i'] = Q['d_i']
        Depart['Event'] = "Depart"
        
        Events = [Arrivals, Depart]
        Events = pd.concat(Events)
        
        #Events = Events.sort_values(by=['t_i'])
        Events = Events.sort_values(['t_i', 'Event'], ascending = (True, False))
        Events.reset_index(level = 0, drop = True, inplace = True)
        
        
        #step through the event list, track how many vehicles are legally parked
        #identify when there is a double parked vehicle and remove it from the
        #event list and record the information so as to calculate total double parking
        total_parked = 0
        dbl_parked_events = pd.DataFrame(columns = ["Truck", "a_i", "d_i"])
        legal_parked_events = pd.DataFrame(columns = ["Truck", "a_i", "d_i"])
        for t in range(0,len(Events)):
            if Events['Event'][t] == 'Arrival':
                total_parked += 1
            elif Events['Event'][t] == 'Depart':
                total_parked += -1
            
            
            #what is the expected departure time of the current event?  Enables checking
            #to see if this truck would exceed the end of the scenario time and
            #therefore should be listed as double parking
            current_truck = Events["Trucks"][t]
            if Events['Event'][t] != 'dbl_park': #having an issue where a 'dbl parked' entry which was previously a depart
                #causes there not to be a current truck departure event
                departure_event = Events.loc[(Events["Trucks"] == current_truck) & (Events["Event"] == "Depart") ].reset_index(drop = True)
                departure = departure_event['t_i']
            else:
                departure = pd.Series(0) #the current entry is a 'dbl park', set departure = 0 so that it is always less than scenario end
                        
                
            if departure.item() > end: #will the current truck departure exceed the end of the scenario?
                #this section of the code may not be necessary anymore becuase we
                #pre-determine the departure time of the DV during Q generation
                #and do not allow any DVs with arrival + service duration that exceeds
                #the end of the scenario to be passed to the FCFS or Optimal schedulers
            
                total_parked += -1
                
                #initialize the collection of data about this double parked vehicle
                #that will be appended to the dbl_parked_events dataframe
                dbl_parked_data = []
                
                #store truck number
                current_truck = Events["Trucks"][t]
                dbl_parked_data.append(current_truck)
                
                #store double parked arrival time
                dbl_parked_data.append(Events["t_i"][t])
                
                #find and store the departure time for the double parked truck
                departure = Events.loc[(Events["Trucks"] == current_truck) & (Events["Event"] == "Depart") ]
                dbl_parked_data.append(departure.iloc[0]["t_i"])
                
                #change the event label as double parking so that it is not counted as a departure
                #from the total parking count
                Events.loc[departure.index.values[0],"Event"] = "dbl_park"
                
                #concatonate the double parking data to the larger double parking events dataframe
                dbl_parked_events.loc[len(dbl_parked_events.index)] = dbl_parked_data
                
                
            elif total_parked > num_spaces: #did the new arrival exceed parking capacity?
                #then the current vehicle is going to double park, b/c it parks for the full duration
                #of service time, remove vehicle from the parking count
                total_parked += -1
                
                #initialize the collection of data about this double parked vehicle
                #that will be appended to the dbl_parked_events dataframe
                dbl_parked_data = []
                
                #store truck number
                current_truck = Events["Trucks"][t]
                dbl_parked_data.append(current_truck)
                
                #store double parked arrival time
                dbl_parked_data.append(Events["t_i"][t])
                
                #find and store the departure time for the double parked truck
                departure = Events.loc[(Events["Trucks"] == current_truck) & (Events["Event"] == "Depart") ]
                dbl_parked_data.append(departure.iloc[0]["t_i"])
         
                #change the event label as double parking so that it is not counted as a departure
                #from the total parking count
                Events.loc[departure.index.values[0],"Event"] = "dbl_park"
                
                #concatonate the double parking data to the larger double parking events dataframe
                dbl_parked_events.loc[len(dbl_parked_events.index)] = dbl_parked_data
                
                
            elif (total_parked <= num_spaces) and (Events['Event'][t] == 'Arrival'): #is the new arrival within parking capacity, then record the parking data
                
                legal_parked_data = []
                
                #store truck number
                current_truck = Events["Trucks"][t]
                legal_parked_data.append(current_truck)
                
                #store double parked arrival time
                legal_parked_data.append(Events["t_i"][t])
                
                #find and store the departure time for the double parked truck
                departure = Events.loc[(Events["Trucks"] == current_truck) & (Events["Event"] == "Depart") ]
                legal_parked_data.append(departure.iloc[0]["t_i"])
                
                #concatonate the legal parking data to the larger legal parking events dataframe
                legal_parked_events.loc[len(legal_parked_events.index)] = legal_parked_data
                
        
        #create a new column that calculcates the service duration which is equivalent
        #to the total time the vehicle is double parked
        dbl_parked_events["s_i"] = dbl_parked_events["d_i"] - dbl_parked_events["a_i"]
        dbl_parked_events['Park Type'] = 'Dbl Park'
        total_dbl_park = sum(dbl_parked_events["s_i"])
        
        #create a new column that calculates the service duration which is equivalent
        #to the total legal parking time
        legal_parked_events["s_i"] = legal_parked_events['d_i'] - legal_parked_events['a_i']
        legal_parked_events['Park Type'] = 'Legal Park'
        
        #combine the dbl park and legal park dataframes into a holistic parking event dataframe
        park_events_FCFS = pd.concat([legal_parked_events, dbl_parked_events])
        park_events_FCFS = park_events_FCFS.sort_values(['a_i'], ascending = True)
        park_events_FCFS.reset_index(level = 0, drop = True, inplace = True)
        
        return total_dbl_park, dbl_parked_events, legal_parked_events, park_events_FCFS

    # except:
    #     return
