# -*- coding: utf-8 -*-
"""
Created on Sun Mar  7 09:12:21 2021

@author: Burns
"""

def MOD_flex(num_trucks, num_spaces, Q, buffer, start, end, end_scenario, t_initialize, x_initialize, prev_obj):
    
    try:
        #added variables to help scale the constraints in the optimization algorithm
        import numpy as np
        scale = end
        scale_s_i = np.mean(Q['s_i'])
        
        import gurobipy as gp
        from gurobipy import GRB
        import numpy as np
        import pandas as pd
        
        # Base data
        
        #M = max(Q['b_i']) + max(Q['s_i']) - min(Q['a_i']) #min a_i effectively becomes
        #zero when we assume that unscheduled or double parked vehicles will be
        #represented by a t_i variable = 0.  If we change the t_0i variable back to
        #the midpoint of a_i and b_i, then we need to add back a_i to the definition
        #of M.  We make the value of M and the bounds of the big M constraint larger
        #in this case, but this may still be more efficient by allowing t_0i = 0.

               
        #Setup vertices and nodes based on requests and Q
        vertices = [i 
                    for i in range(0, len(Q)+1)]
        
        arcs = [(i, j) 
                for i in range(0, len(vertices)) 
                for j in range(0, len(vertices)) if j != i]
        
        
        #Generate the specific M for each i and jointData vehicle pair.  Of note, this code
        #will generate the diagonal elements of the matrix as well, e.g. i == jointData, but
        #this will not impact the contraints below because there is logic to prevent
        #generating a constraint where i == jointData
        
        M_df = pd.DataFrame()
        M_inst = []
        for i in range(0, len(Q)):
            for j in range(0, len(Q)):
                  M_ij = min(end_scenario, Q['b_i'].iloc[i] + Q['phi'].iloc[i] + Q['s_i'].iloc[i] + buffer) - max(start, Q['a_i'].iloc[j] - Q['phi'].iloc[j])
                  M_inst.append(M_ij)
                
            M_df[i] = M_inst
            M_inst = []
        
        M_df = M_df.transpose()
        
        #M = max(Q['b_i']) + max(Q['s_i']) - min(Q['a_i'])
              
        #-----------------------------------------------------------------------------
        # Create optimization model
        m = gp.Model('smartcurb')
        m.setParam('OutputFlag', 0)
        m.update()
        
        #-----------------------------------------------------------------------------
        # VARIABLES
        
        #create the start time decision variable, based on start time for Truck_1
        t_i = m.addVars(Q.shape[0], vtype = GRB.CONTINUOUS, name = 't_i')
        #ub = end,
        
        #create arc active decision variable
        x_i_j = m.addVars(arcs, vtype = GRB.BINARY, name = 'x_i_j')
        
        #create indicator variable if request does not meet time window MOD4b
        #b_i = m.addVars(Q.shape[0], vtype = GRB.BINARY, name = 'b_i')
        #ub = end, 
        
        m.update()
        
        #-----------------------------------------------------------------------------
        # CONSTRAINTS
        
        # restricts the number of routes starting from the depot ? (2)
        m.addConstr(x_i_j.sum(0, '*') <= num_spaces)
        
        # Flow balance (3) ??
        m.addConstrs((x_i_j.sum(i+1, '*') - x_i_j.sum('*', i+1)) == 0 
                      for i in range(0, len(Q)))
        
        # Requests should either be allocated or not allocated to a parking space
        for i in range(0, len(Q)):
            if Q.iloc[i]['Prev Assigned'] != 'Yes': #e.g. could be 'No' or nan
                m.addConstr(x_i_j.sum(i+1, '*') <= 1)
            elif Q.iloc[i]['Prev Assigned'] == 'Yes':
                if Q.iloc[i]['a_i'] == start: #is this vehicle a hold over from the prev optimization window, then make sure to start a parking sequence with this vehicle
                    m.addConstr(x_i_j[0,i+1] == 1)
                else: #this vehicle is prev scheduled, but not starting until after the start of the window, or was not a hold over from a prev optimization
                    m.addConstr(x_i_j.sum(i+1, '*') == 1)
                
        # m.addConstrs(x_i_j.sum(i+1, '*') <= 1 
        #       for i in range(0, len(Q)))
        # + b_i[i] 
        
        # Relate time and service time to flow (5)
        #m.addConstrs(t_i[jointData] >= ( (t_i[i] + Q['s_i'][i] + buffer) - (1 - x_i_j[i+1, jointData+1])*M )
        #m.addConstrs(t_i[i] + Q['s_i'][i] + buffer - t_i[jointData] <= (1 - x_i_j[i+1, jointData+1])*M
        
        # m.addConstrs(t_i[jointData] >= t_i[i] + Q['s_i'][i] + buffer - (1 - x_i_j[i+1, jointData+1])*M
        #               for i in range(0, len(Q)) 
        #               for jointData in range(0, len(Q)) if jointData != i)
        
        m.addConstrs( (t_i[j]/scale) >= ((t_i[i] + Q['s_i'].iloc[i] + buffer - ((1 - x_i_j[i+1, j+1])*M_df.iloc[i, j]))/scale) #go back to *M if going with the generic original M and not M_ij, M_df.iloc[i, jointData]
                      for i in range(0, len(Q)) 
                      for j in range(0, len(Q)) if j != i)
        
        
        #time window constraints (6) REMOVED for MOD4b
        #m.addConstrs(t_i[i] >= Q['a_i'][i] for i in t_i)
        #m.addConstrs(t_i[i] <= Q['b_i'][i] for i in t_i)
        
        # scenario time window (7)
        m.addConstrs( ((t_i[i] + Q['s_i'].iloc[i] + buffer)/scale) <= (end_scenario/scale) for i in t_i )
        m.addConstrs( (t_i[i]/scale) >= (start/scale) for i in t_i)
        m.addConstrs( (t_i[i]/scale) <= (end/scale) for i in t_i)
        
        #Earliness/Tardiness Constraints for MOD4b (24, 25)
        #m.addConstrs(K[0]*b_i[i] >= Q['a_i'][i] - t_i[i] - flex for i in b_i)
        #m.addConstrs(K[0]*b_i[i] >= t_i[i] - Q['b_i'][i] - flex for i in b_i)
        
        #Replace Earliness/Tardiness Constraints from MOD4b + (also remove b_i)
        for i in t_i:
            #t_0i = int(round((Q['a_i'][i] + Q['b_i'][i]) / 2))
            #t_0i = 0 #t_start
            #t_0i = Q['a_i'].iloc[i]#/np.mean(Q['a_i']) #first paper version
            if Q['a_i'].iloc[i] < start: #can have a conflict where we are considering a_i that occurred before the start of this window 
                t_0i = start
            elif Q['a_i'].iloc[i] >= start:
                t_0i = Q['a_i'].iloc[i]
                
            m.addConstr( (t_i[i]/scale) >= (Q['a_i'].iloc[i] - Q['phi'].iloc[i] - ((1-x_i_j.sum(i+1, '*'))*(Q['a_i'].iloc[i] - Q['phi'].iloc[i] - t_0i)) )/scale)
            m.addConstr( (t_i[i]/scale) <= (Q['b_i'].iloc[i] + Q['phi'].iloc[i] + ((1-x_i_j.sum(i+1, '*'))*(t_0i - Q['b_i'].iloc[i] - Q['phi'].iloc[i])) )/scale)
        
        #new constraint to prevent schedule trucks from being scheduled after 720 minutes
        #if the vehicle is cancelled then this constraint is met and not applicable
        #m.addConstrs((1-b_i[i])*t_i[i] <= end for i in b_i)
        #possible linear version of the above nonlinear formulation
        #m.addConstrs(t_i[i] <= (1-b_i[i])*end for i in b_i)
        
        #new constraint to assign unscheduled vehicle start time based on original
        #time window
        #m.addConstrs(t_i[i] <= Q['b_i'][i] + b_i[i]*(((Q['a_i'][i] + Q['b_i'][i]) / 2) - Q['b_i'][i]) for i in b_i)
        #m.addConstrs(t_i[i] >= Q['a_i'][i] - b_i[i]*(Q['a_i'][i] - ((Q['a_i'][i] + Q['b_i'][i]) / 2)) for i in b_i)
        
        
        #New constraint to ensure that the total minutes of unscheduled service
        #is greater than or equal to the objective function from the previous
        #optimization, the pure PAP optimization
        total_min_dbl_parked = gp.LinExpr()
        for i in range(0, len(Q)):
            total_min_dbl_parked += (1-x_i_j.sum(i+1, '*'))*(Q['s_i'].iloc[i]/scale_s_i)
            
        m.addConstr(total_min_dbl_parked <= prev_obj)
        
        #-----------------------------------------------------------------------------
        #Objective Function
        
        #have trouble implementing the min(abs(x)) where x = t_i[i] - Q['a_i'].iloc[i]
        #so, following these instruction to work around and create a linear function
        # https://support.gurobi.com/hc/en-us/community/posts/360074428532-How-to-handle-absolute-value-in-objective-function-
        # https://math.stackexchange.com/questions/432003/converting-absolute-value-program-into-linear-program
        r_i = m.addVars(Q.shape[0], vtype = GRB.CONTINUOUS, name = 'r_i')
        s_i = m.addVars(Q.shape[0], vtype = GRB.CONTINUOUS, name = 's_i')
        m.update()
        
        assignment_dev = gp.LinExpr()
        for i in range(0, len(Q)):
            x = (x_i_j.sum(i+1, '*'))*(t_i[i]/scale - Q['a_i'].iloc[i]/scale) #deviation weight by if the truck was assigned or not (only consider deviation if truck is assigned)
            m.addConstr(r_i[i] - s_i[i] == x)

            assignment_dev += (r_i[i] + s_i[i])
        
                
        m.setObjective(assignment_dev, GRB.MINIMIZE)
        
        #bound the objective function to be greater than zero
        m.addConstr(assignment_dev >= 0)
        
        #-----------------------------------------------------------------------------
        #Set Model Parameters
        
        m.setParam('TimeLimit', 45)
        #print(m.getParamInfo('TimeLimit'))
        
        # # Symmetry Detection https://www.gurobi.com/documentation/9.1/refman/symmetry.html#parameter:Symmetry
        # # -1 = default; 0 = off; 1 = concervative, 2 = aggressive
        #m.setParam('Symmetry', 2)
        
        # # MIP Focus https://www.gurobi.com/documentation/9.1/refman/mipfocus.html#parameter:MIPFocus
        # # 1 = find feasible solns quickly, 2 = focus on proving the optimal soln, 3 = focus on the bound if obj bound moving slowly
        # m.setParam('MIPFocus', 3)
        
        # # Determines the amount of time spent in MIP heuristics
        # m.setParam('Heuristics', .5)
        
        # # Controls the cutting planes
        # # -1 = auto; 0 = none; 1 = conservative; 2 = aggressive
        # m.setParam('Cuts', 2)
        
        #The default settings of these parameters resulted in inaccurate optimal solutions
        #"Warning: max contraint violation (4.7995e-06) exceeds tolerance".  I explored
        #different values of the parameters and found that as I increased the value of each individually
        #I could get the error to go away.  So, as a next best option, I set all of the tolerances
        #to a larger value than default.  This may induce additional error in the optimal solution
        #but the error should be minimal and should allow the solver to create an accurate optimal solution.
        #Combination of best judgement and empirical experimentation for picking the best parameter values.
        #m.setParam('IntFeasTol', 0.01)
        #m.setParam('FeasibilityTol', 0.00001)
        #m.setParam('OptimalityTol', 0.01)
        
        #m.setParam('MIPGapAbs', 1)
        m.setParam('MIPGap', .01)
        
        #m.setParam('Method', 4) #https://www.gurobi.com/documentation/9.5/refman/method.html
        
        #-----------------------------------------------------------------------------
        #Set initial conditions from First Come First Serve solution
        #if (flex != 5): #added, switch from 0 to 5
        for item in t_i:
            #print(Q['t_i'][i])
            #t_i[i].start = Q['t_i'][i]
            t_i[item].start = t_initialize[item]
            
        #if (flex != 5): #added, switch from 0 to 5 (0 originally)
        i = 0
        for item in x_i_j:
            x_i_j[item].start = x_initialize[i]
            i += 1
        
        #for i in range(0, len(cancelled)):
            #print(cancelled[i])
            #b_i[i].start = cancelled[i]
        
        m.update()
        
        #-----------------------------------------------------------------------------
        # Compute optimal solution
        m.optimize()
        print('\n')
        m.printQuality()
        
        
        # for i in t_i:
        #     print(t_i[i])
        #     print((1-x_i_j.sum(i+1, '*')).getValue())
        # print('\n')
        
        #if the PAP is infeasible, possibly due to an inability to find a solution
        #in the required processing time, return empty outputs so that the algoritm
        #can continue into the AP formulation.  This may mask errors in the PAP,
        #e.g. constraint errors, and therefore it will be important to check the
        #PAPvAP data to make sure the PAP is still being processed
        if m.status == 9:
            
            return m.status, None, None, None, None, None, None

        #----------------------------------------------------------------------
        #create output statistics
        count_b_i = 0
        #social_cost = []
        flex_dbl_park = 0
                        
        
        for i in t_i:
            if (1-x_i_j.sum(i+1, '*')).getValue() == 1:
                #print(b_i[i])
                count_b_i += 1


    
        #commented to faster processing
        if m.status == GRB.OPTIMAL:
            
            #if there are any non-scheduled vehicles ("cancellations") then add
            #the total service time to time double parked
            for i in t_i:
                if (1-x_i_j.sum(i+1, '*')).getValue() == 1:
                    flex_dbl_park += Q['s_i'].iloc[i]

           
            for i in x_i_j:
                if x_i_j[i].getAttr('x') > 0.99:
                    print(x_i_j[i])
                    
            #for i in x_i_j:
                #print(x_i_j[i])
                
            for i in t_i:
                if (1-x_i_j.sum(i+1, '*')).getValue() == 1:
                    print(str(t_i[i]) + ' is unscheduled (add 1 to index value for truck in Q)')
                else:
                    print(t_i[i])


        #store data on the double parked vehicles, e.g. when did they start double
        #parking and for how long
        dbl_park_events = pd.DataFrame(columns = ["Vehicle", "a_i", "s_i", "d_i"])
        
        for i in t_i:
            dbl_parked_data = []
            if (1-x_i_j.sum(i+1, '*')).getValue() == 1:
                dbl_parked_data.append(Q['Vehicle'].iloc[i])
                dbl_parked_data.append(Q['a_i'].iloc[i])
                dbl_parked_data.append(Q['s_i'].iloc[i])
                dbl_parked_data.append(Q['d_i'].iloc[i])
            
                dbl_park_events.loc[len(dbl_park_events.index)] = dbl_parked_data

        #store data on all of the vehicles, legally parked and double parked
        park_events = pd.DataFrame(columns = ['Vehicle', 'a_i', 's_i', 'd_i', 'Park Type'])
        for i in t_i:
            #print(i)
            park_data = []
            #double parked delivery vehicles
            if np.round((1-x_i_j.sum(i+1, '*')).getValue()) == 1:
                park_data.append(Q['Vehicle'].iloc[i])
                park_data.append(Q['a_i'].iloc[i])
                park_data.append(Q['s_i'].iloc[i])
                park_data.append(Q['d_i'].iloc[i])
                park_data.append('No Park')
            elif np.round((1-x_i_j.sum(i+1, '*')).getValue()) == 0: #legally parked delivery vehicles
                park_data.append(Q['Vehicle'].iloc[i])
                park_data.append(np.round(t_i[i].getAttr("x")))
                park_data.append(Q['s_i'].iloc[i])
                park_data.append(np.round(t_i[i].getAttr("x")) + Q['s_i'].iloc[i])
                park_data.append('Legal Park')
                
            park_events.loc[len(park_events.index)] = park_data
            
        park_events = park_events.sort_values(['a_i'], ascending = True)
        park_events.reset_index(level = 0, drop = True, inplace = True)

        #pull the solution values from t_i for use in the next solver/phi iteration
        end_state_t_i = []
        for i in t_i:
            end_state_t_i.append(t_i[i].getAttr("x"))
            
        end_state_x_ij = []
        for i in x_i_j:
            end_state_x_ij.append(int(x_i_j[i].getAttr("x")))
            
        total_min_dbl_parked_soln = total_min_dbl_parked.getValue()

        print('LOL Objective = {}'.format(m.getObjective()))
                    
        return m.status, m.getObjective().getValue()*scale, count_b_i, end_state_t_i, end_state_x_ij, dbl_park_events, park_events, total_min_dbl_parked_soln
    

        
    
    except:
        return


def MOD_flex_cf(num_trucks, num_spaces, Q, buffer, start, end, end_scenario, t_initialize, x_initialize, prev_obj):
    # try:
        # added variables to help scale the constraints in the optimization algorithm
        import numpy as np
        scale = end
        scale_s_i = np.mean(Q['s_i'])


        scale_n_i = np.mean(Q['n_i'])
        scale_p_i = np.mean(Q['p_i'])


        if (scale_p_i == 0):
            scale_p_i = 1
        if (scale_n_i == 0):
            scale_n_i = 1

        import gurobipy as gp
        from gurobipy import GRB
        import numpy as np
        import pandas as pd

        # Base data

        # M = max(Q['b_i']) + max(Q['s_i']) - min(Q['a_i']) #min a_i effectively becomes
        # zero when we assume that unscheduled or double parked vehicles will be
        # represented by a t_i variable = 0.  If we change the t_0i variable back to
        # the midpoint of a_i and b_i, then we need to add back a_i to the definition
        # of M.  We make the value of M and the bounds of the big M constraint larger
        # in this case, but this may still be more efficient by allowing t_0i = 0.

        # Setup vertices and nodes based on requests and Q
        vertices = [i
                    for i in range(0, len(Q) + 1)]

        arcs = [(i, j)
                for i in range(0, len(vertices))
                for j in range(0, len(vertices)) if j != i]

        # Generate the specific M for each i and jointData vehicle pair.  Of note, this code
        # will generate the diagonal elements of the matrix as well, e.g. i == jointData, but
        # this will not impact the contraints below because there is logic to prevent
        # generating a constraint where i == jointData

        M_df = pd.DataFrame()
        M_inst = []
        for i in range(0, len(Q)):
            for j in range(0, len(Q)):
                M_ij = min(end_scenario, Q['b_i'].iloc[i] + Q['phi'].iloc[i] + Q['s_i'].iloc[i] + buffer) - max(start,
                                                                                                                Q[
                                                                                                                    'a_i'].iloc[
                                                                                                                    j] -
                                                                                                                Q[
                                                                                                                    'phi'].iloc[
                                                                                                                    j])
                M_inst.append(M_ij)

            M_df[i] = M_inst
            M_inst = []

        M_df = M_df.transpose()

        # M = max(Q['b_i']) + max(Q['s_i']) - min(Q['a_i'])

        # -----------------------------------------------------------------------------
        # Create optimization model
        m = gp.Model('smartcurb')
        m.setParam('OutputFlag', 0)
        m.update()
        # -----------------------------------------------------------------------------
        # VARIABLES

        # create the start time decision variable, based on start time for Truck_1
        t_i = m.addVars(Q.shape[0], vtype=GRB.CONTINUOUS, name='t_i')
        # ub = end,

        # create arc active decision variable
        x_i_j = m.addVars(arcs, vtype=GRB.BINARY, name='x_i_j')

        # create indicator variable if request does not meet time window MOD4b
        # b_i = m.addVars(Q.shape[0], vtype = GRB.BINARY, name = 'b_i')
        # ub = end,

        m.update()

        # -----------------------------------------------------------------------------
        # CONSTRAINTS

        # restricts the number of routes starting from the depot ? (2)
        m.addConstr(x_i_j.sum(0, '*') <= num_spaces)

        # Flow balance (3) ??
        m.addConstrs((x_i_j.sum(i + 1, '*') - x_i_j.sum('*', i + 1)) == 0
                     for i in range(0, len(Q)))

        # Requests should either be allocated or not allocated to a parking space
        for i in range(0, len(Q)):
            if Q.iloc[i]['Prev Assigned'] != 'Yes':  # e.g. could be 'No' or nan
                m.addConstr(x_i_j.sum(i + 1, '*') <= 1)
            elif Q.iloc[i]['Prev Assigned'] == 'Yes':
                if Q.iloc[i][
                    'a_i'] == start:  # is this vehicle a hold over from the prev optimization window, then make sure to start a parking sequence with this vehicle
                    m.addConstr(x_i_j[0, i + 1] == 1)
                else:  # this vehicle is prev scheduled, but not starting until after the start of the window, or was not a hold over from a prev optimization
                    m.addConstr(x_i_j.sum(i + 1, '*') == 1)

        # m.addConstrs(x_i_j.sum(i+1, '*') <= 1
        #       for i in range(0, len(Q)))
        # + b_i[i]

        # Relate time and service time to flow (5)
        # m.addConstrs(t_i[jointData] >= ( (t_i[i] + Q['s_i'][i] + buffer) - (1 - x_i_j[i+1, jointData+1])*M )
        # m.addConstrs(t_i[i] + Q['s_i'][i] + buffer - t_i[jointData] <= (1 - x_i_j[i+1, jointData+1])*M

        # m.addConstrs(t_i[jointData] >= t_i[i] + Q['s_i'][i] + buffer - (1 - x_i_j[i+1, jointData+1])*M
        #               for i in range(0, len(Q))
        #               for jointData in range(0, len(Q)) if jointData != i)

        m.addConstrs((t_i[j] / scale) >= (
                    (t_i[i] + Q['s_i'].iloc[i] + buffer - ((1 - x_i_j[i + 1, j + 1]) * M_df.iloc[i, j])) / scale)
                     # go back to *M if going with the generic original M and not M_ij, M_df.iloc[i, jointData]
                     for i in range(0, len(Q))
                     for j in range(0, len(Q)) if j != i)

        # time window constraints (6) REMOVED for MOD4b
        # m.addConstrs(t_i[i] >= Q['a_i'][i] for i in t_i)
        # m.addConstrs(t_i[i] <= Q['b_i'][i] for i in t_i)

        # scenario time window (7)
        m.addConstrs(((t_i[i] + Q['s_i'].iloc[i] + buffer) / scale) <= (end_scenario / scale) for i in t_i)
        m.addConstrs((t_i[i] / scale) >= (start / scale) for i in t_i)
        m.addConstrs((t_i[i] / scale) <= (end / scale) for i in t_i)

        # Earliness/Tardiness Constraints for MOD4b (24, 25)
        # m.addConstrs(K[0]*b_i[i] >= Q['a_i'][i] - t_i[i] - flex for i in b_i)
        # m.addConstrs(K[0]*b_i[i] >= t_i[i] - Q['b_i'][i] - flex for i in b_i)

        # Replace Earliness/Tardiness Constraints from MOD4b + (also remove b_i)
        for i in t_i:
            # t_0i = int(round((Q['a_i'][i] + Q['b_i'][i]) / 2))
            # t_0i = 0 #t_start
            # t_0i = Q['a_i'].iloc[i]#/np.mean(Q['a_i']) #first paper version
            if Q['a_i'].iloc[
                i] < start:  # can have a conflict where we are considering a_i that occurred before the start of this window
                t_0i = start
            elif Q['a_i'].iloc[i] >= start:
                t_0i = Q['a_i'].iloc[i]

            m.addConstr((t_i[i] / scale) >= (Q['a_i'].iloc[i] - Q['phi'].iloc[i] - (
                        (1 - x_i_j.sum(i + 1, '*')) * (Q['a_i'].iloc[i] - Q['phi'].iloc[i] - t_0i))) / scale)
            m.addConstr((t_i[i] / scale) <= (Q['b_i'].iloc[i] + Q['phi'].iloc[i] + (
                        (1 - x_i_j.sum(i + 1, '*')) * (t_0i - Q['b_i'].iloc[i] - Q['phi'].iloc[i]))) / scale)

        # new constraint to prevent schedule trucks from being scheduled after 720 minutes
        # if the vehicle is cancelled then this constraint is met and not applicable
        # m.addConstrs((1-b_i[i])*t_i[i] <= end for i in b_i)
        # possible linear version of the above nonlinear formulation
        # m.addConstrs(t_i[i] <= (1-b_i[i])*end for i in b_i)

        # new constraint to assign unscheduled vehicle start time based on original
        # time window
        # m.addConstrs(t_i[i] <= Q['b_i'][i] + b_i[i]*(((Q['a_i'][i] + Q['b_i'][i]) / 2) - Q['b_i'][i]) for i in b_i)
        # m.addConstrs(t_i[i] >= Q['a_i'][i] - b_i[i]*(Q['a_i'][i] - ((Q['a_i'][i] + Q['b_i'][i]) / 2)) for i in b_i)

        # New constraint to ensure that the total minutes of unscheduled service
        # is greater than or equal to the objective function from the previous
        # optimization, the pure PAP optimization
        new_obj = gp.LinExpr()
        for i in range(0, len(Q)):
            new_obj += (1 - x_i_j.sum(i + 1, '*')) * (Q['n_i'].iloc[i] / scale_n_i)
            new_obj += (- x_i_j.sum(i + 1, '*')) * (Q['p_i'].iloc[i] / scale_p_i)

        m.addConstr(new_obj <= prev_obj)

        # -----------------------------------------------------------------------------
        # Objective Function

        # have trouble implementing the min(abs(x)) where x = t_i[i] - Q['a_i'].iloc[i]
        # so, following these instruction to work around and create a linear function
        # https://support.gurobi.com/hc/en-us/community/posts/360074428532-How-to-handle-absolute-value-in-objective-function-
        # https://math.stackexchange.com/questions/432003/converting-absolute-value-program-into-linear-program
        r_i = m.addVars(Q.shape[0], vtype=GRB.CONTINUOUS, name='r_i')
        s_i = m.addVars(Q.shape[0], vtype=GRB.CONTINUOUS, name='s_i')
        m.update()

        assignment_dev = gp.LinExpr()
        for i in range(0, len(Q)):
            x = (x_i_j.sum(i + 1, '*')) * (t_i[i] / scale - Q['a_i'].iloc[
                i] / scale)  # deviation weight by if the truck was assigned or not (only consider deviation if truck is assigned)
            m.addConstr(r_i[i] - s_i[i] == x)

            assignment_dev += (r_i[i] + s_i[i])

        m.setObjective(assignment_dev, GRB.MINIMIZE)

        # bound the objective function to be greater than zero
        m.addConstr(assignment_dev >= 0)

        # -----------------------------------------------------------------------------
        # Set Model Parameters

        m.setParam('TimeLimit', 45)
        # print(m.getParamInfo('TimeLimit'))

        # # Symmetry Detection https://www.gurobi.com/documentation/9.1/refman/symmetry.html#parameter:Symmetry
        # # -1 = default; 0 = off; 1 = concervative, 2 = aggressive
        # m.setParam('Symmetry', 2)

        # # MIP Focus https://www.gurobi.com/documentation/9.1/refman/mipfocus.html#parameter:MIPFocus
        # # 1 = find feasible solns quickly, 2 = focus on proving the optimal soln, 3 = focus on the bound if obj bound moving slowly
        # m.setParam('MIPFocus', 3)

        # # Determines the amount of time spent in MIP heuristics
        # m.setParam('Heuristics', .5)

        # # Controls the cutting planes
        # # -1 = auto; 0 = none; 1 = conservative; 2 = aggressive
        # m.setParam('Cuts', 2)

        # The default settings of these parameters resulted in inaccurate optimal solutions
        # "Warning: max contraint violation (4.7995e-06) exceeds tolerance".  I explored
        # different values of the parameters and found that as I increased the value of each individually
        # I could get the error to go away.  So, as a next best option, I set all of the tolerances
        # to a larger value than default.  This may induce additional error in the optimal solution
        # but the error should be minimal and should allow the solver to create an accurate optimal solution.
        # Combination of best judgement and empirical experimentation for picking the best parameter values.
        # m.setParam('IntFeasTol', 0.01)
        # m.setParam('FeasibilityTol', 0.00001)
        # m.setParam('OptimalityTol', 0.01)

        # m.setParam('MIPGapAbs', 1)
        m.setParam('MIPGap', .01)

        # m.setParam('Method', 4) #https://www.gurobi.com/documentation/9.5/refman/method.html

        # -----------------------------------------------------------------------------
        # Set initial conditions from First Come First Serve solution
        # if (flex != 5): #added, switch from 0 to 5
        for item in t_i:
            # print(Q['t_i'][i])
            # t_i[i].start = Q['t_i'][i]
            t_i[item].start = t_initialize[item]

        # if (flex != 5): #added, switch from 0 to 5 (0 originally)
        i = 0
        for item in x_i_j:
            x_i_j[item].start = x_initialize[i]
            i += 1

        # for i in range(0, len(cancelled)):
        # print(cancelled[i])
        # b_i[i].start = cancelled[i]

        m.update()

        # -----------------------------------------------------------------------------
        # Compute optimal solution
        m.optimize()
        print('\n')
        m.printQuality()

        # for i in t_i:
        #     print(t_i[i])
        #     print((1-x_i_j.sum(i+1, '*')).getValue())
        # print('\n')

        # if the PAP is infeasible, possibly due to an inability to find a solution
        # in the required processing time, return empty outputs so that the algoritm
        # can continue into the AP formulation.  This may mask errors in the PAP,
        # e.g. constraint errors, and therefore it will be important to check the
        # PAPvAP data to make sure the PAP is still being processed
        if m.status == 9:
            return m.status, None, None, None, None, None, None

        # ----------------------------------------------------------------------
        # create output statistics
        count_b_i = 0
        # social_cost = []
        flex_dbl_park = 0

        for i in t_i:
            if (1 - x_i_j.sum(i + 1, '*')).getValue() == 1:
                # print(b_i[i])
                count_b_i += 1

        # commented to faster processing
        if m.status == GRB.OPTIMAL:

            # if there are any non-scheduled vehicles ("cancellations") then add
            # the total service time to time double parked
            for i in t_i:
                if (1 - x_i_j.sum(i + 1, '*')).getValue() == 1:
                    flex_dbl_park += Q['s_i'].iloc[i]

            for i in x_i_j:
                if x_i_j[i].getAttr('x') > 0.99:
                    print(x_i_j[i])

            # for i in x_i_j:
            # print(x_i_j[i])

            for i in t_i:
                if (1 - x_i_j.sum(i + 1, '*')).getValue() == 1:
                    print(str(t_i[i]) + ' is unscheduled (add 1 to index value for truck in Q)')
                else:
                    print(t_i[i])

        # store data on the double parked vehicles, e.g. when did they start double
        # parking and for how long
        dbl_park_events = pd.DataFrame(columns=["Vehicle", "a_i", "s_i", "d_i"])

        for i in t_i:
            dbl_parked_data = []
            if (1 - x_i_j.sum(i + 1, '*')).getValue() == 1:
                dbl_parked_data.append(Q['Vehicle'].iloc[i])
                dbl_parked_data.append(Q['a_i'].iloc[i])
                dbl_parked_data.append(Q['s_i'].iloc[i])
                dbl_parked_data.append(Q['d_i'].iloc[i])

                dbl_park_events.loc[len(dbl_park_events.index)] = dbl_parked_data

        # store data on all of the vehicles, legally parked and double parked
        park_events = pd.DataFrame(columns=['Vehicle', 'a_i', 's_i', 'd_i', 'Park Type'])
        for i in t_i:
            # print(i)
            park_data = []
            # double parked delivery vehicles
            if np.round((1 - x_i_j.sum(i + 1, '*')).getValue()) == 1:
                park_data.append(Q['Vehicle'].iloc[i])
                park_data.append(Q['a_i'].iloc[i])
                park_data.append(Q['s_i'].iloc[i])
                park_data.append(Q['d_i'].iloc[i])
                park_data.append('No Park')
            elif np.round((1 - x_i_j.sum(i + 1, '*')).getValue()) == 0:  # legally parked delivery vehicles
                park_data.append(Q['Vehicle'].iloc[i])
                park_data.append(np.round(t_i[i].getAttr("x")))
                park_data.append(Q['s_i'].iloc[i])
                park_data.append(np.round(t_i[i].getAttr("x")) + Q['s_i'].iloc[i])
                park_data.append('Legal Park')

            park_events.loc[len(park_events.index)] = park_data

        park_events = park_events.sort_values(['a_i'], ascending=True)
        park_events.reset_index(level=0, drop=True, inplace=True)

        # pull the solution values from t_i for use in the next solver/phi iteration
        end_state_t_i = []
        for i in t_i:
            end_state_t_i.append(t_i[i].getAttr("x"))

        end_state_x_ij = []
        for i in x_i_j:
            end_state_x_ij.append(int(x_i_j[i].getAttr("x")))

        new_obj_soln = new_obj.getValue()

        return m.status, m.getObjective().getValue() * scale, count_b_i, end_state_t_i, end_state_x_ij, dbl_park_events, park_events, new_obj_soln




    # except:
    #     return
                    