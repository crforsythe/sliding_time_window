import pandas as p
import numpy as np
from SimulateNHTSData import load_nhts_data, select_n_trips, construct_truth_dataframe, gen_truck_arrivals, join_requests
from copy import deepcopy
import gurobipy as gp
from gurobipy import GRB

#This function creates a new instance of a Gurboi model with the name specified by the "name" argument
def createGurobiModel(name='smartCurb'):
    m = gp.Model(name)
    m.setParam('OutputFlag', 0)
    m.update()
    return m

#This function takes in the overarching model and then creates all of the time variables necessary for scheduling
def createTimeVars(model, data):
    #Specifies a continuous variable for the model for each vehicle with lower and upper bound constraints to prevent
    #from being assigned outside of a given time window [a_i_OG, b_i_OG]
    t_i = model.addVars(data.shape[0], vtype=GRB.CONTINUOUS, name='t_i', lb=data.loc[:, 'a_i_OG'], ub=data.loc[:, 'b_i_OG'])
    model.update()
    return model, t_i

#This function creates all of the flow variables necessary for problem construction
#Additionally, for cases when i=jointData, a constraint requiring those entries to be zero
#This is slightly different than Aaron's implementation but will lead to an easier
#account of x_i_j variables
def createFlowVars(model, data):
    #Construct all posible i,jointData combinations
    arcs = [(i, j)
            for i in range(len(data)+1)
            for j in range(len(data)+1)]
    #Add all arcs as indicies for the X matrix; binary
    x_i_j = model.addVars(arcs, vtype=GRB.BINARY, name='x_i_j')

    for i in range(len(data)):
        model.addConstr(x_i_j[i,i]==0, name='diag-{}'.format(i))

    model.update()
    return model, x_i_j

#This function creates a constraint that limits the number of parking spaces considered
def createNumberParkingSpotConstraint(model, x_i_j, num_spaces=1):
    model.addConstr(x_i_j.sum(0, '*') <= num_spaces, name='NumSpaces')
    return model, x_i_j

#This function creates a set of constraints that preserve flow across X matrix decision variable
def createFlowPreservationConstraints(model, x_i_j, data):
    for i in range(len(data)):
        model.addConstr((x_i_j.sum(i + 1, '*') - x_i_j.sum('*', i + 1)) == 0, name='FlowPres-{}'.format(i))

    model.update()
    return model, x_i_j


#This function generates and return the Big M matrix for access throughout all other functions
def getBigMMatrix(data, start_scenario=0, end_scenario=(60*24), buffer=5):
    M_df = p.DataFrame()
    M_inst = []
    for i in range(0, len(data)):
        for j in range(0, len(data)):
            M_ij = min(end_scenario, data['b_i_OG'].iloc[i] + data['s_i'].iloc[i] + buffer) - max(start_scenario, data['a_i_OG'].iloc[j])
            M_inst.append(M_ij)

        M_df[i] = M_inst
        M_inst = []

    M_df = M_df.transpose()


    return M_df

#This function creates a set of constraints that prevents overlap in time variables
def createTimeOverlapConstraints(model, t_i, x_i_j, data, start_scenario=0, end_scenario=(24*60), buffer=5, scale=1):
    #Get the big M values for each vehicle pairing
    M_df = getBigMMatrix(data, start_scenario, end_scenario, buffer)

    for i in range(len(data)):
        for j in range(len(data)):
            if(i!=j):
                model.addConstr((t_i[j] / scale) >= (
            (t_i[i] + data['s_i'].iloc[i] + buffer - ((1 - x_i_j[i + 1, j + 1]) * M_df.iloc[i, j])) / scale), name='TimeOverlap; {}-{}'.format(i, j))

    model.update()

    return model, t_i, x_i_j

#This function creates all of the constraints to prevent times being assigned outside the considered scenario
def createTimeWindowConstraints(model, t_i, data, start_scenario=0, end_scenario=(24*60), buffer=5, scale=1):

    for i in t_i:

        model.addConstr(((t_i[i] + data['s_i'].iloc[i] + buffer) / scale) <= (end_scenario / scale), name='ParkPast-{}'.format(i))
        model.addConstr((t_i[i] / scale) >= (start_scenario / scale), name='Start-{}'.format(i))
        model.addConstr((t_i[i] / scale) <= (end_scenario / scale), name='End-{}'.format(i))

    model.update()

    return model, t_i

#Appies the constraints that prevent times being shifted outside of the [a_i, b_i] domain
def createTimeShiftConstraints(model, t_i, x_i_j, data, start_scenario=0, end_scenario=(24*60), buffer=5, scale=1):
    for i in t_i:
        if(data['a_i_OG'].iloc[i] < start_scenario):  # can have a conflict where we are considering a_i that occurred before the start of this window
            t_0i = start_scenario
        elif(data['a_i_OG'].iloc[i] >= start_scenario):
            t_0i = data['a_i_OG'].iloc[i]

        m.addConstr((t_i[i] / scale) >= (data['a_i_OG'].iloc[i] - ((1 - x_i_j.sum(i + 1, '*')) * (data['a_i_OG'].iloc[i] - t_0i))) / scale, name='StartShift - {}'.format(i))
        m.addConstr((t_i[i] / scale) <= (data['b_i_OG'].iloc[i] + ((1 - x_i_j.sum(i + 1, '*')) * (t_0i - data['b_i_OG'].iloc[i]))) / scale, name='EndShift - {}'.format(i))

    m.update()

    return model, t_i, x_i_j

#Edits the models params.
def setModelParams(model, **kwargs):
    for key, val in kwargs.items():
        model.setParam(key, val)

    model.update()
    return model

#Return double park expression for setting of objective
def getDoubleParkExpression(data, x_i_j, scale=1):
    obj = gp.LinExpr()
    for i in range(0, len(data)):
        obj += (1-x_i_j.sum(i+1, '*'))*(data['s_i'].iloc[i]/scale)

    return obj

def setModelObjective(model, *args):
    for obj in args:
        model.setObjective(obj, GRB.MINIMIZE)

    model.update()
    return model

np.random.seed(8131970)
data = load_nhts_data()
t_sample = select_n_trips(data, num_sample=10)
t = construct_truth_dataframe(t_sample)

truck = gen_truck_arrivals(10)

jointData = join_requests(t, truck)
jointData.loc[:, 'b_i_OG'] = jointData.loc[:, 'a_i_OG']+10
w = 50
objective_dict = {'parked': [0, 0], 'weights': [w, 100-w], 'cols': [['Expected Double Park', 's_i'], ['Expected Cruising', 'Expected Cruising Time']]}
# req_master = run_optimization(deepcopy(t), objective_dict, buffer=0)

m = createGurobiModel()
m, t_i = createTimeVars(m, jointData)
m, x_i_j = createFlowVars(m, jointData)
m, x_i_j = createNumberParkingSpotConstraint(m, x_i_j, num_spaces=4)
m, x_i_j = createFlowPreservationConstraints(m, x_i_j, jointData)
bigM = getBigMMatrix(jointData)
m, t_i, x_i_j = createTimeOverlapConstraints(m, t_i, x_i_j, jointData)
m, t_i = createTimeWindowConstraints(m, t_i, jointData)
m, t_i, x_i_j = createTimeShiftConstraints(m, t_i, x_i_j, jointData)
m = setModelParams(m, TimeLimit=45, MIPGap=0.01)
doubleParkObj = getDoubleParkExpression(jointData, x_i_j)
m = setModelObjective(m, doubleParkObj)
t0 = m.status
m.optimize()
t = m.status, m.getObjective().getValue()

end_state_t_i = []
for i in t_i:
    end_state_t_i.append(t_i[i].getAttr("x"))

end_state_x_ij = []
for i in range(len(jointData)+1):
    for j in range(len(jointData) + 1):
        end_state_x_ij.append(int(x_i_j[i, j].getAttr("x")))