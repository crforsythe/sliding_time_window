# -*- coding: utf-8 -*-
"""
Created on Wed Sep 22 09:29:14 2021

@author: Burns
"""

def gen_veh_arrivals(max_trucks, end=2400):
    
    import numpy as np
    import pandas as pd
    
    #np.random.seed(442)
    
    a_i = []
    s_i = []
    
    
    # random draw of arrival time
    
    x = np.random.uniform(0, 1, max_trucks)
    #x = [.999, .02]
    # from empirical Coord data determine which starting hour the random draw
    # cooresponds to
    emp_total = 1520
    s_flag_idx = []
    
    
    std_dev = 5 #assumed std dev for the service duration random variable
    
    for i in x:
        
        #this flag is active until the random start_window + service duration is
        #verified to not be longer than the end of the scenario.  If the sum is
        #larger than the end of the scenario, then the script will redraw new 
        #values and go through the verification process again
        flag = True
        
        while flag == True:
            
            if i <= 56 / emp_total:
                # start time is 7am, but this cooresponds to 0 minutes in the model
                start_window = 0
                serv_dur = int(np.random.normal(62, std_dev))

            elif i <= (56 + 118) / emp_total:
                start_window = 60
                serv_dur = int(np.random.normal(45, std_dev))

            elif i <= (56 + 118 + 190) / emp_total:
                start_window = 120
                serv_dur = int(np.random.normal(40, std_dev))

            elif i <= (56 + 118 + 190 + 285) / emp_total:
                start_window = 180
                serv_dur = int(np.random.normal(41, std_dev))
       
            elif i <= (56 + 118 + 190 + 285 + 267) / emp_total:
                start_window = 240
                serv_dur = int(np.random.normal(32, std_dev))
           
            elif i <= (56 + 118 + 190 + 285 + 267 + 218) / emp_total:
                start_window = 300
                serv_dur = int(np.random.normal(30, std_dev))
   
            elif i <= (56 + 118 + 190 + 285 + 267 + 218 + 182) / emp_total:
                start_window = 360
                serv_dur = int(np.random.normal(31, std_dev))
         
            elif i <= (56 + 118 + 190 + 285 + 267 + 218 + 182 + 127) / emp_total:
                start_window = 420
                serv_dur = int(np.random.normal(27, std_dev))
      
            elif i <= (56 + 118 + 190 + 285 + 267 + 218 + 182 + 127 + 55) / emp_total:
                start_window = 480
                serv_dur = int(np.random.normal(25, std_dev))

            elif i <= (56 + 118 + 190 + 285 + 267 + 218 + 182 + 127 + 55 + 19) / emp_total:
                start_window = 540
                serv_dur = int(np.random.normal(31, std_dev))
    
            elif i <= (56 + 118 + 190 + 285 + 267 + 218 + 182 + 127 + 55 + 19 + 3) / emp_total:
                start_window = 600
                serv_dur = int(np.random.normal(24, std_dev))
      
                                
            #randomly draw 1 minute within an hour to add to the arrival hour
            start_window += int(np.random.uniform(0, 60))
            
           #check to see if the current start_window + serv_dur exceeds the end
           #of the scenario time, and redraw if this is the case
            if start_window + serv_dur <= end:
                flag = False #the current random draw is acceptable, change the 
                #flag, exit the while loop and proceed to the next randow draw
            else:
                i = np.random.uniform(0, 1) #the current random draw is not acceptable
                #redraw a random number and determine the start_window and service
                #duration again
            
        #after we have a set of start_window and service duration that do not 
        #exceed the end of the scenario, store those values
        a_i.append(start_window)
        s_i.append(serv_dur)
        
        start_window = None
        serv_dur = None
        
    
    #create a dictionary from lists of data that will be turned into a dataframe
    requests = {'a_i_OG': a_i, 'b_i_OG': a_i, 's_i': s_i}
    
    #create index label for Q
    Trucks = ['Truck_' + str(i+1) 
          for i in range(0, len(requests['a_i_OG']))]
    
    Q = pd.DataFrame(requests, index = Trucks)
    
    Q['d_i_OG'] = Q['a_i_OG'] + Q['s_i']
    
    #return the sum of the service durations for the instance
    sum_s = sum(s_i)
    Q.loc[:, ['a_i_OG', 'b_i_OG', 'd_i_OG']]+=(7*60)
    Q.loc[:, 'Type'] = 'Truck'
    
    return Q


    
