# -*- coding: utf-8 -*-
"""
Created on Tue Nov  2 11:38:27 2021

@author: Burns
"""


import gurobipy as gp
from gurobipy import GRB
import numpy as np
import pandas as pd


def AP(num_trucks, end, Q, num_spaces, bids, flex, buffer, x_initialize):
    
    x = [(i, j) 
            for i in range(1, num_trucks +1) #based on the number of trucks
            for j in range(0, end +1)] #based on the number of minutes in the scenario
    
    
    
    #-----------------------------------------------------------------------------
    # Create optimization model
    m = gp.Model('smartcurb_AP')
    
    #-----------------------------------------------------------------------------
    # DECISION VARIABLES
    
    x_i_j = m.addVars(x, vtype = GRB.BINARY, name = 'x_i_j')
    
    m.update()
    
    
    #-----------------------------------------------------------------------------
    # CONSTRAINTS
    
    # Constraint 4.38 
    m.addConstrs((x_i_j.sum(i, '*')) <= 1
                 for i in range(1, num_trucks +1))
    
    
    # Constraint 4.39 
    # There is an error in T_ij from the Yang et al paper, in that either
    # t_k <= t_j or t_k >= t_j-s_i.  It isn't obvious based on the logic below
    # which expression is being implemented, but after stepping through the 
    # constraints that are generated, t_k <= t_j has been implemented.  This
    # formulation creates one additional constraint that includes start times at
    # t_end.  This is possible in the current formulation as x_ij is defined through
    # x_i_end.  Worst case we weren't supposed to model a start time at the end
    # and this formulation creates one additional constraint that isn't utilized / 
    # is redudant.  The optimal solution is correct, but there might be a very small
    # increase in runtime.
    for j in range(0, end +1):
        #print("jointData: " + str(jointData))
        constr = gp.LinExpr()
        for i in range(1, num_trucks +1):
            #print("i: " + str(i))
            
            if j >= Q['s_i'][i-1] + buffer: #add -1 for big scenario
                k_lower = int(j - Q['s_i'][i-1] - buffer +1) #add -1 for big scenario
                k_upper = int(j +1)
                for k in range(k_lower, k_upper):
                    constr += x_i_j[i, k]
            else:            
                k_lower = 0
                k_upper = j +1
                for k in range(k_lower, k_upper):
                    constr += x_i_j[i, k] 
        
            #print(jointData, i)
            
        m.addConstr(constr <= num_spaces)
        #print('jointData: ' + str(jointData))
        #print(constr) 
            
            
    # New constraint which prevents delivery vehicles from being scheduled at a
    # time when their service duration would exceed the end of the scenario
    for i in range(1, num_trucks +1):
        #initialize a new constraint for each delivery vehicle
        constr = gp.LinExpr()
        #determine the last possible start time for each delivery vehicle, need
        #to remove the service time and buffer
        latest_start = end - Q['s_i'][i-1] - buffer
        
        #step through the window of the unavailable start times and create a sumation
        #of the decision variables
        for j in range(latest_start, end +1):
            constr += x_i_j[i,j]
            
        m.addConstr(constr == 0)
            
    #-----------------------------------------------------------------------------
    #Set initial conditions from First Come First Serve solution
    
    #method for solely running the AP
    # if flex != 0:
    #     i = 0
    #     for item in x_i_j:
    #         x_i_j[item].start = x_initialize[i]
    #         i += 1
            
    #method for running both PAP and AP
    if x_initialize != [None]:
        i = 0
        for item in x_i_j:
            x_i_j[item].start = x_initialize[i]
            i += 1
    
    #for i in range(0, len(cancelled)):
        #print(cancelled[i])
        #b_i[i].start = cancelled[i]
        
    m.update()            
    
    
    #-----------------------------------------------------------------------------
    #Objective Function
    
    # expr = gp.quicksum(b_i)
    # m.setObjective(expr, GRB.MINIMIZE)
    
    
    
    obj = gp.LinExpr()
    for i in range(1, num_trucks +1):
        for j in range(0, end +1):
            #obj += Q['s_i'][i-1]*x_i_j[i,jointData]
            obj += bids.iloc[j][i -1]*x_i_j[i,j]
    #     obj += b_i[i]*Q['s_i'][i]
    #     #obj = b_i*weight + e_i*weights?
    
    m.setObjective(obj, GRB.MAXIMIZE)
    
    #-----------------------------------------------------------------------------
    #Set Model Parameters
    
    m.setParam('TimeLimit', 120)
    
    # # MIP Focus https://www.gurobi.com/documentation/9.1/refman/mipfocus.html#parameter:MIPFocus
    # # 1 = find feasible solns quickly, 2 = focus on proving the optimal soln, 3 = focus on the bound if obj bound moving slowly
    m.setParam('MIPFocus', 3)
    
    m.setParam('MIPGap', .01)
    
    # Compute optimal solution
    m.optimize()
    print('\n')
    
    # print('Total Service Time: ')
    # print(sum(Q['s_i']))
    # print('\n')
    
    
    
    if m.status == GRB.OPTIMAL:
        #iterate through the decision variables and determine which are equal to "1"
        # or represent the start time of a vehicle
        for i in x_i_j:
            if x_i_j[i].getAttr('x') == 1:
                print(x_i_j[i])
                
        end_state_x_i_j = []
        for i in x_i_j:
            end_state_x_i_j.append(int(x_i_j[i].getAttr("x")))
                
    
    
    #store data on the double parked vehicles, e.g. when did they start double
    #parking and for how long
    dbl_park_events = pd.DataFrame(columns = ["Truck", "a_i", "s_i", "d_i"])
    
    for i in range(0, len(Q)):
        dbl_parked_data = []
        if x_i_j.sum(i+1, '*').getValue() == 0:
                        
            dbl_parked_data.append(Q["Trucks"][i])
            dbl_parked_data.append(Q['a_i'][i])
            dbl_parked_data.append(Q['s_i'][i])
            dbl_parked_data.append(Q['d_i'][i])
        
            dbl_park_events.loc[len(dbl_park_events.index)] = dbl_parked_data
            
    
    #store data on all of the vehicle, legally parked and double parked
    #first though, need to assess the x_i_j to find when vehicle i is scheduled, captured by jointData
    park_events = pd.DataFrame(columns = ['Truck', 'a_i', 's_i', 'd_i', 'Park Type'])
    for i in range(0, len(Q)):
        park_data = []
        park_data.append(Q['Trucks'][i])
        #test to see if the truck is double parking
        if np.round(x_i_j.sum(i+1, '*').getValue()) == 0:
            park_data.append(Q['a_i'][i])
            park_data.append(Q['s_i'][i])
            park_data.append(Q['a_i'][i] + Q['s_i'][i])
            park_data.append("Dbl Park")
        else:
            for j in range(0, end):
                #search over time step jointData for the assignment (==1)
                #if int(x_i_j[i+1, jointData].getAttr("x")) == 1:
                if np.round(x_i_j[i+1, j].getAttr("x")) == 1:
                    park_data.append(j)
                    park_data.append(Q['s_i'][i])
                    park_data.append(j + Q['s_i'][i])
                    park_data.append("Legal Park")
                    break
                
        park_events.loc[len(park_events.index)] = park_data
        
    park_events = park_events.sort_values(['a_i'], ascending = True)
    park_events.reset_index(level = 0, drop = True, inplace = True)
    
    
    
    park_demand = sum(Q['s_i'])
    print('Total Parking Demand: ' + str(park_demand))
    print('Optimal Parking Satisfied: ' + str(m.getObjective().getValue()))
    dbl_park_Opt = park_demand - m.getObjective().getValue()
    print('Total dbl_parking: ' + str(dbl_park_Opt))
    
    
    return m.status, m.getObjective().getValue(), dbl_park_Opt, park_demand, end_state_x_i_j, dbl_park_events, park_events
    