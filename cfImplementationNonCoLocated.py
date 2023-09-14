import pandas as p
import numpy as np
from SimulateNHTSData import load_nhts_data, select_n_trips, construct_truth_dataframe, gen_truck_arrivals, join_requests
from copy import deepcopy
import gurobipy as gp
from gurobipy import GRB
from collections import OrderedDict
from tqdm import tqdm, trange
from datetime import datetime
from gurobipy import abs_ as gabs
import pickle
import multiprocessing as mp
# from execute_v2 import park_events_FCFS,runtime_FCFS
from seq_arrival_new import seq_curb
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

    for i in range(len(data)+1):
        model.addConstr(x_i_j[i,i]==0, name='diag-{}'.format(i))

    model.update()
    return model, x_i_j


def getLocationCols(data):
    locationCols = []

    for col in data.columns:
        if ('Location' in col):
            locationCols.append(col)

    return locationCols


def getLocationColInds(data):
    locationColInds = []

    for col in data.columns:
        if ('Location' in col):
            locationColInds.append(list(data.columns).index(col))

    return locationColInds


#Creates the location decision variables as well as their baseline constraints
def createLocationVars(model, data, numSpaces = (1,1)):
    locationCols = getLocationCols(data)
    locationColInds = getLocationColInds(data)

    vehLocs = [(i,j) for i in range(len(data))
               for j in range(len(locationCols))]

    p_i_k = model.addVars(vehLocs, vtype=GRB.BINARY, name='p_i_k')

    for i in range(len(data)):
        k = 0
        model.addConstr(p_i_k.sum(i, '*')<=1,name='p_{}_k<=1'.format(i))
        for j in locationColInds:
            model.addConstr(p_i_k[i,k]<=data.iloc[i,j], name='p_{}_{}<Preference'.format(i,k))
            p_i_k[i,k].Start = 0
            k+=1

    model.update()



    return model, p_i_k

def createLocationSlackVars(model, data, numSpots = (1,1)):
    locationCols = getLocationCols(data)
    locationColInds = getLocationColInds(data)

    vehLocs = [(i,j) for i in range(len(data))
               for j in range(len(locationCols))]

    y_i_k = model.addVars(vehLocs, vtype=GRB.BINARY, name='y_i_k')

    for k in range(len(locationCols)):
        try:
            model.addConstr(y_i_k.sum('*', k)<=numSpots[k])
        except:
            print('NO AMOUNT OF SPOTS SPECIFIED FOR PARKING SPOT {}'.format(k))
            model.addConstr(y_i_k.sum('*', k) <= numSpots[k])
    model.update()
    return model, y_i_k

#This function creates a constraint that limits the number of parking spaces considered
def createNumberParkingSpotConstraint(model, x_i_j, num_spaces=1):
    model.addConstr(x_i_j.sum(0, '*') <= np.sum(num_spaces), name='NumSpaces')
    model.update()
    return model, x_i_j

#This function creates a set of constraints that preserve flow across X matrix decision variable
def createFlowPreservationConstraints(model, x_i_j, data):
    for i in range(len(data)+1):
        model.addConstr((x_i_j.sum(i, '*') - x_i_j.sum('*', i)) == 0, name='FlowPres-{}'.format(i))

    model.update()
    return model, x_i_j


def createLocationFlowPreservationConstraints(model, x_i_j, p_i_k, data):
    locationCols = getLocationCols(data)


    for i in range(len(data)):
        for j in range(len(data)):
            for k in range(len(locationCols)):
                model.addConstr((p_i_k[i,k]-p_i_k[j,k])<=(1-x_i_j[i+1,j+1]), name='LocFlowLess-{}'.format((i,j,k)))
                model.addConstr((p_i_k[i, k] - p_i_k[j, k]) >= (x_i_j[i + 1, j + 1]-1), name='LocFlowGreater-{}'.format((i,j,k)))

    model.update()

    return model, x_i_j, p_i_k

#Applies constraint requiring that each vehicle is only assigned to a parking spot once
def createSingleAssignmentConstraints(model, x_i_j, data):
    for i in range(1, len(data)+1):
        model.addConstr((x_i_j.sum(i, '*')) <= 1, name='SingleAssignment-{}'.format(i))

    model.update()
    return model, x_i_j

def createSingleLocationConstraints(model, x_i_j, p_i_k, data):

    for i in range(len(data)):
        model.addConstr(p_i_k.sum(i, '*')==x_i_j.sum(i+1, '*'), name='SingleLocationAssignment-{}'.format(i))

    model.update()

    return model, x_i_j, p_i_k


#This function generates and return the Big M matrix for access throughout all other functions
def getBigMMatrix(data, start_scenario=0, end_scenario=(60*24), buffer=0):

    M_inst = []
    for i in range(0, len(data)):
        tempRow = []
        for j in range(0, len(data)):
            M_ij = min(end_scenario, data['b_i_OG'].iloc[i] + data['s_i'].iloc[i] + buffer) - max(start_scenario, data['a_i_OG'].iloc[j])
            tempRow.append(1e8)

        M_inst.append(tempRow)
    M_df = p.DataFrame(M_inst)


    return M_df

#This function creates a set of constraints that prevents overlap in time variables
def createTimeOverlapConstraints(model, t_i, x_i_j, data, start_scenario=0, end_scenario=(24*60), buffer=0, scale=1):
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
def createTimeWindowConstraints(model, t_i, data, start_scenario=0, end_scenario=(24*60), buffer=0, scale=1):

    for i in range(len(data)):

        # model.addConstr(((t_i[i] + data['s_i'].iloc[i] + buffer) / scale) <= (end_scenario / scale), name='ParkPast-{}'.format(i)) #TODO This constraint is causing infeasible solutions and really isn't necessary
        model.addConstr((t_i[i] / scale) >= (start_scenario / scale), name='Start-{}'.format(i))
        model.addConstr((t_i[i] / scale) <= (end_scenario / scale), name='End-{}'.format(i))

    model.update()

    return model, t_i

#Appies the constraints that prevent times being shifted outside of the [a_i, b_i] domain
def createTimeShiftConstraints(model, t_i, x_i_j, data, start_scenario=0, end_scenario=(24*60), buffer=0, scale=1):
    for i in t_i:
        if(data['a_i_OG'].iloc[i] < start_scenario):  # can have a conflict where we are considering a_i that occurred before the start of this window
            t_0i = start_scenario
        elif(data['a_i_OG'].iloc[i] >= start_scenario):
            t_0i = data['a_i_OG'].iloc[i]

        model.addConstr((t_i[i] / scale) >= (data['a_i_OG'].iloc[i] - ((1 - x_i_j.sum(i + 1, '*')) * (data['a_i_OG'].iloc[i] - t_0i))) / scale, name='StartShift - {}'.format(i))
        model.addConstr((t_i[i] / scale) <= (data['b_i_OG'].iloc[i] + ((1 - x_i_j.sum(i + 1, '*')) * (t_0i - data['b_i_OG'].iloc[i]))) / scale, name='EndShift - {}'.format(i))

    model.update()

    return model, t_i, x_i_j

#Constructs constraints that ensure that all previous decisions regarding time are respected
def createPersistentTimeDecisionConstraints(model, t_i, data):
    i = 0
    for index, row in data.iterrows():

        if(row['Assigned']==1):
            model.addConstr(t_i[i]==row['a_i_Final'], name='Prev-T-{}'.format(index))
            t_i[i].Start = row['a_i_Final']
        i+=1
    return model, t_i

#Constructs constraints that ensure that all previous decisions regarding flow are respected
def createPersistentFlowDecisionConstraints(model, x_i_j, prev_x_i_j):

    for i in range(len(prev_x_i_j)):
        for j in range(len(prev_x_i_j)):
    #         if(j==0):
    #             #Previous decisions with flow heading back to the "depot" need not be respected as
    #             #The flow could move to a newly-added vehicle
    #             model.addConstr(x_i_j[i,j] <= prev_x_i_j[i,j], name='Prev-X-{}-{}'.format(i,j))
    #         else:
    #             model.addConstr(x_i_j[i, j] == prev_x_i_j[i, j], name='Prev-X-{}-{}'.format(i, j))
            x_i_j[i,j].Start = prev_x_i_j[i, j]


    for i in range(1, len(prev_x_i_j)):
        model.addConstr(x_i_j.sum(i, '*')==np.sum(prev_x_i_j[i, :]), name='Prev-X-{}'.format(i))


    return model, x_i_j

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

def getExpectedDoubleParkExpression(data, x_i_j, scale=1):
    obj = gp.LinExpr()
    for i in range(0, len(data)):
        obj += (1-x_i_j.sum(i+1, '*'))*(data['s_i'].iloc[i]*data['Expected Double Park'].iloc[i]/scale)

    return obj

def getExpectedCruisingExpression(data, x_i_j, scale=1):
    obj = gp.LinExpr()
    for i in range(0, len(data)):
        obj += (1 - x_i_j.sum(i + 1, '*')) * (data['Expected Cruising Time'].iloc[i] * data['Expected Cruising'].iloc[i] / scale)

    return obj


def applyDeviationObjectiveVariables(model, t_i, x_i_j, data, scale=1):
    # have trouble implementing the min(abs(x)) where x = t_i[i] - Q['a_i'].iloc[i]
    # so, following these instruction to work around and create a linear function
    # https://support.gurobi.com/hc/en-us/community/posts/360074428532-How-to-handle-absolute-value-in-objective-function-
    # https://math.stackexchange.com/questions/432003/converting-absolute-value-program-into-linear-program
    # r_i = model.addVars(data.shape[0], vtype=GRB.CONTINUOUS, name='r_i')
    # s_i = model.addVars(data.shape[0], vtype=GRB.CONTINUOUS, name='s_i')


    # model.update()
    #
    # assignment_dev = gp.LinExpr()
    # for i in range(0, len(data)):
    #     x = (x_i_j.sum(i + 1, '*')) * (t_i[i] / scale - data['a_i_OG'].iloc[
    #         i] / scale)  # deviation weight by if the truck was assigned or not (only consider deviation if truck is assigned)
    #     model.addConstr(r_i[i] - s_i[i] == x)
    #
    #     assignment_dev += (r_i[i] + s_i[i])
    #

    assignment_dev = gp.LinExpr()
    for i in range(0, len(data)):
        assignment_dev+=(t_i[i] - data['a_i_OG'].iloc[i])

    model.setObjectiveN(assignment_dev, 50, 0, 1, name='AbsDev')
    model.update()

    return model, t_i, x_i_j

#Returns the objective function associated with double-parking
def setModelObjective(model, *args):
    for obj in args:
        model.setObjective(obj, GRB.MINIMIZE)
    # if(type(args[0])==list):
    #
    # else:
    #     for obj in args:
    #         model.setObjectiveN(obj[0], GRB.MINIMIZE, priority=obj[1], weight=obj[2])

    model.update()
    return model

def setModelObjectiveN(model, obj, index, priority, weight):

    model.setObjectiveN(obj, index, priority, weight)
    model.update()
    return model

#Returns a pretty version of the optimization output
def constructOutputs(model, t_i, x_i_j, p_i_k, y_i_k, data, code=-1):
    # Outputs of Interest
    # t = model.status, model.getObjective().getValue()
    locationCols = getLocationCols(data)


    end_state_t_i = []
    for i in t_i:
        end_state_t_i.append(t_i[i].getAttr("x"))

    end_state_t_i = np.array(end_state_t_i)

    end_state_x_ij = []
    for i in range(len(data) + 1):
        tempRow = []
        for j in range(len(data) + 1):
            tempRow.append(int(x_i_j[i, j].getAttr("x")))
        end_state_x_ij.append(tempRow)

    end_state_x_ij = np.array(end_state_x_ij)

    end_state_p_ik = []
    for i in range(len(data)):
        tempRow = []
        for k in range(len(locationCols)):
            tempRow.append(int(p_i_k[i, k].getAttr("x")))
        end_state_p_ik.append(tempRow)

    end_state_p_ik = np.array(end_state_p_ik)

    end_state_y_ik = []
    for i in range(len(data)):
        tempRow = []
        for k in range(len(locationCols)):
            tempRow.append(int(y_i_k[i, k].getAttr("x")))
        end_state_y_ik.append(tempRow)

    end_state_y_ik = np.array(end_state_y_ik)

    r = {'t': end_state_t_i, 'x': end_state_x_ij, 'p':end_state_p_ik, 'y':end_state_y_ik, 'data':data, 'code':code}

    return r

def createLocationSlackConstraints(model, x_i_j, p_i_k, y_i_k, data):
    locationCols = getLocationCols(data)

    for i in range(len(data)):
        for k in range(len(locationCols)):
            model.addConstr(y_i_k[i,k]>=x_i_j[0, i+1]+p_i_k[i,k]-1, name='Location Slack - {}'.format((i,k)))

    return model, x_i_j, p_i_k, y_i_k

def applyAssignedTimesToData(returnDict):
    # print(returnDict)
    cleanOutputs = constructOutputs(*list(returnDict.values()))
    cleanOutputs['data'].loc[:, 'a_i_Final'] = cleanOutputs['t']
    cleanOutputs['data'].loc[:, 'd_i_Final'] = cleanOutputs['data'].loc[:, 'a_i_Final']+cleanOutputs['data'].loc[:, 's_i']
    cleanOutputs['data'].loc[:, 'Assigned'] = np.sum(cleanOutputs['x'], axis=1)[1:]
    return cleanOutputs

def getUnchangingIndicies(outcomeList, currentTime):

    #Take out data for easy access
    data0 = outcomeList[-2]['data']
    data1 = outcomeList[-1]['data']

    # Collect shared indices
    index0 = set(list(data0.index))
    index1 = set(list(data1.index))

    sharedIndices = index0 & index1

    if(len(sharedIndices)==0):
        return None

    #Store binary identification of shared indicies across data
    sharedIndices0 = data0.index.isin(sharedIndices)
    sharedIndices1 = data1.index.isin(sharedIndices)

    #Create separate set of shared indices for x, since x also includes the depot
    sharedXIndices0 = [True]
    sharedXIndices1 = [True]

    sharedXIndices0.extend(sharedIndices0)
    sharedXIndices1.extend(sharedIndices1)


    #Get relevant x outcomes and the difference across iterations
    x0 = outcomeList[-2]['x']
    x0 = x0[sharedXIndices0, :]
    x0 = x0[:, sharedXIndices0]
    x1 = outcomeList[-1]['x']
    x1 = x1[sharedXIndices1, :]
    x1 = x1[:, sharedXIndices1]

    xDiff = x1-x0
    # Get relevant t outcomes and the difference across iterations
    t0 = outcomeList[-2]['t'][sharedIndices0]
    t1 = outcomeList[-1]['t'][sharedIndices1]

    tDiff = t1 - t0



    #Indices that are finished parking
    preIndices = data1.loc[sharedIndices, 'd_i_Final']<currentTime
    #If things remain the same between time steps and a given vehicle is no
    #longer relevant they can be dropped. Inverse is also stored.
    indicesToDrop = (tDiff==0)&(preIndices)&(xDiff[1:, :].sum(axis=1)==0)
    indicesToKeep = np.logical_not(indicesToDrop)

    xIndicesToDrop = [True]
    xIndicesToKeep = [True]

    xIndicesToDrop.extend(list(indicesToDrop))
    xIndicesToKeep.extend(list(indicesToKeep))

    r = {}
    r['locked'] = {}
    r['unlocked'] = {}

    r['locked']['t'] = t1[list(indicesToDrop)]
    if(len(r['locked']['t']>0)):
        r['locked']['x'] = x1
        r['locked']['x'] = r['locked']['x'][xIndicesToDrop, :]
        r['locked']['x'] = r['locked']['x'][:, xIndicesToDrop]
        # r['locked']['xFull'] = x1
        # r['locked']['tFull'] = t1
    else:
        r['locked']['x'] = np.array([])
    # r['locked']['data'] = data1.loc[indicesToDrop.index, :]
    r['locked']['indices'] = list(indicesToDrop[indicesToDrop==True].index)

    r['unlocked']['indices'] = list(data1.loc[~data1.index.isin(r['locked']['indices']), :].index)
    r['unlocked']['t'] = outcomeList[-1]['t'][data1.index.isin(r['unlocked']['indices'])]
    r['unlocked']['data'] = outcomeList[-1]['data'].loc[r['unlocked']['indices'], :]
    r['unlocked']['x'] = outcomeList[-1]['x']
    xUnlockedIndices = [True]
    xUnlockedIndices.extend(list(data1.index.isin(r['unlocked']['indices'])))

    r['unlocked']['x'] = r['unlocked']['x'][xUnlockedIndices, :]
    r['unlocked']['x'] = r['unlocked']['x'][:, xUnlockedIndices]
    if(len(r['locked']['t']>0)):
        r['unlocked']['x'][0, :] = outcomeList[-1]['x'][:, xUnlockedIndices].sum(axis=0)-np.multiply(r['unlocked']['x'][1:, :].sum(axis=0), 1)
        r['unlocked']['x'][0, 0] = 0

    return r


def runSlidingOptimization(data, numSpaces=(1,1), tau=5, start=0, stop=(24*60)+1, buffer=0, weightDoubleParking=1, weightCruising=0, timeLimit=45):
    # Model construction and optimization
    #r is a list that will store successive results over time
    #Data should be sorted by received so that constraints moving from one time
    #step to the next are not assigned to the wrong vehicles
    if(np.mean(data.loc[:, 'Received'])<0):
        data = data.sort_values('Received_OG')
    else:
        data = data.sort_values('Received')
    data.index = range(len(data))
    r = []
    unchangingDicts = []
    indiciesToDrop = []
    iterVar = trange(max(start, min(data.loc[:, 'Received'])), stop, tau)
    passPrevDecisions = False
    for j in iterVar:

        if (len(r) > 1):
            unchangingDict = getUnchangingIndicies(r, j)
            if(unchangingDict!=None):
                unchangingDicts.append(unchangingDict)
                indiciesToDrop.extend(unchangingDict['locked']['indices'])
        data = data.loc[~data.index.isin(indiciesToDrop), :]
        tempData = data.loc[data.loc[:, 'Received'] <= j, :]

        if(len(tempData)>0): #
            #Create Model
            m = createGurobiModel()
            m, t_i = createTimeVars(m, tempData)
            m, x_i_j = createFlowVars(m, tempData)
            m, p_i_k = createLocationVars(m, tempData, numSpaces=numSpaces)
            m, y_i_k = createLocationSlackVars(m, tempData, numSpaces)

            # Now set constraints on previous decisions should there be any
            # Changes over the running time
            # Drop vehicle considered whenever answers haven't changed
            if (len(r) > 1 and unchangingDict!=None):
                m, t_i = createPersistentTimeDecisionConstraints(m, t_i, unchangingDict['unlocked']['data'])
                m, x_i_j = createPersistentFlowDecisionConstraints(m, x_i_j, unchangingDict['unlocked']['x'])
            elif (len(r) > 0 and passPrevDecisions):
                m, t_i = createPersistentTimeDecisionConstraints(m, t_i, r[-1]['data'])
                m, x_i_j = createPersistentFlowDecisionConstraints(m, x_i_j, r[-1]['x'])

            m, x_i_j = createNumberParkingSpotConstraint(m, x_i_j, num_spaces=numSpaces)
            m, x_i_j = createFlowPreservationConstraints(m, x_i_j, tempData)
            m, x_i_j, p_i_k = createLocationFlowPreservationConstraints(m, x_i_j, p_i_k, tempData)
            m, x_i_j = createSingleAssignmentConstraints(m, x_i_j, tempData)
            m, x_i_j, p_i_k = createSingleLocationConstraints(m, x_i_j, p_i_k, tempData)
            m, x_i_j, p_i_k, y_i_k = createLocationSlackConstraints(m, x_i_j, p_i_k, y_i_k, tempData)
            m, t_i, x_i_j = createTimeOverlapConstraints(m, t_i, x_i_j, tempData, buffer=buffer)
            m, t_i = createTimeWindowConstraints(m, t_i, tempData)
            m, t_i, x_i_j = createTimeShiftConstraints(m, t_i, x_i_j, tempData)
            # m = setModelParams(m, TimeLimit=timeLimit, MIPGap=0.01, ModelSense=GRB.MINIMIZE, Threads=1)
            m = setModelParams(m, TimeLimit=timeLimit, MIPGap=0.0001, ModelSense=GRB.MINIMIZE, Threads=1)
            doubleParkObj = getExpectedDoubleParkExpression(tempData, x_i_j)
            cruisingObj = getExpectedCruisingExpression(data, x_i_j)
            m = setModelObjectiveN(m, doubleParkObj, 0, 1, weightDoubleParking)
            m = setModelObjectiveN(m, cruisingObj, 1, 1, weightCruising)
            m, t_i, x_i_j = applyDeviationObjectiveVariables(m, t_i, x_i_j, tempData)
            # absDifferenceObj = getAbsoluteDeviationExpression(data, t_i)
            # m = setModelObjective(m, [[doubleParkObj, 100, 1], [absDifferenceObj, 0, 1]])



            t0 = m.status
            m.optimize()
            m.update()
            print('Termination Code: {}'.format(m.status))

            if(m.status==2):
                tempR = OrderedDict({'model':m, 't':t_i, 'x':x_i_j, 'p':p_i_k, 'y':y_i_k, 'data':tempData})
                tempR = applyAssignedTimesToData(tempR)
            else:
                tempR = m.status
                tempR = OrderedDict({'model': m, 't': t_i, 'x': x_i_j, 'data': tempData})
                tempR['data'].loc[:, 'StatusCode'] = m.status
                tempR = applyAssignedTimesToData(tempR)

            tempR['indiciesToDrop'] = list(set(indiciesToDrop))
            tempR['status'] = m.status
            tempR['Current Time'] = j
            m.update()
            r.append(tempR)


    r = collateFinalOutcomes(r)

    try:
        r.loc[:, 'Unassigned'] = 1 - r.loc[:, 'Assigned']
        r = r.sort_values('Received')
        r = r.sort_values(['Unassigned', 'a_i_Final'])
    except:
        pass
    return r

def getLastIndexFromPreviousOutcome(currentOutcome, prevOutcome):
    if(prevOutcome==None):
        return None
    else:
        currentVehicles = list(currentOutcome['data'].loc[:, 'Vehicle'])
        prevVehicles = list(prevOutcome['data'].loc[:, 'Vehicle'])

        if(currentVehicles==prevVehicles):
            return None


        firstVehicle = currentVehicles[0]
        try:
            r = prevVehicles.index(firstVehicle)
        except:
            r = -1

        return r


def getRelevantOutcomes(outcome, index, back=True):
    xVals = outcome['x']
    tVals = outcome['t']
    dataVals = outcome['data']

    r = {}

    if(index==-1):
        r['x'] = xVals[1:, 1:]
        r['t'] = tVals
        r['data'] = dataVals
        r['depotVert'] = xVals[1:, 0]
        r['depotHoriz'] = xVals[0, 1:]

    elif(not back):
        r['x'] = xVals[index:, index:]
        r['t'] = tVals[index:]
        r['data'] = dataVals.iloc[index:, :]
        r['depotVert'] = xVals[index:, 0]
        r['depotHoriz'] = xVals[0, index:]
    else:
        r['x'] = xVals[1:index, 1:index]
        r['t'] = tVals[:index]
        r['data'] = dataVals.iloc[:index, :]
        r['depotVert'] = xVals[1:index, 0]
        r['depotHoriz'] = xVals[0, 1:index]

    r['status'] = outcome['status']

    if(r['status']!=2):
        print("WARNING - NON-OPTIMAL SOLUTION BEING ADDED TO FINAL OUTCOMES")

    return r

def collateFinalOutcomes(finalOutcomes):
    count = 0
    datas = []
    i = 0
    for i in range(1, len(finalOutcomes)):
        prevOutcome = finalOutcomes[i - 1]
        outcome = finalOutcomes[i]

        prevData = prevOutcome['data']
        data = outcome['data']

        prevVehicles = set(prevData.loc[:, 'Vehicle'])
        vehicles = set(data.loc[:, 'Vehicle'])

        lostVehicles = prevVehicles - vehicles
        indiciesToDrop = outcome['indiciesToDrop']

        # count+=len(outcome['data'])
        datas.append(prevData.loc[prevData.loc[:, 'Vehicle'].isin(lostVehicles), :])
        # datas.append(prevData.loc[indiciesToDrop, :])

    try:
        datas.append(outcome['data'])
        datas = p.concat(datas)

        return datas
    except:
        return finalOutcomes[0]['data']


def testFunc(x, y):
    return x**4+y

def testFunc2(x):
    return x**4

def collateTimeWindowOutcomes(outcomeList):

    i = 0

    prevOutcome = None
    finalOutcomes = []
    for outcome in outcomeList:
        lastIndex = getLastIndexFromPreviousOutcome(outcome, prevOutcome)
        if(lastIndex!=None and lastIndex!=0):
            finalOutcomes.append(getRelevantOutcomes(prevOutcome, lastIndex))
        prevOutcome = outcome
        i+=1

    finalOutcomes.append(getRelevantOutcomes(outcome, lastIndex, back=False))

    r = collateFinalOutcomes(finalOutcomes)

    return outcomeList, r


def simulateData(numCars, numTrucks, nhts_data, windowShift=10, receivedDelta=120):
    data = nhts_data
    r = OrderedDict()
    if(numCars==0):
        t_sample = select_n_trips(data, num_sample=1)  # Change the number of personal vehicles simulated
    else:
        t_sample = select_n_trips(data, num_sample=numCars)  # Change the number of personal vehicles simulated

    t = construct_truth_dataframe(t_sample, receivedDelta=receivedDelta)

    if(numTrucks==0):
        truck = gen_truck_arrivals(1)  # Change the number of trucks simulated
    else:
        truck = gen_truck_arrivals(numTrucks)  # Change the number of trucks simulated

    jointData = join_requests(t, truck, receivedDelta=receivedDelta)
    jointData.loc[:, 'b_i_OG'] = jointData.loc[:, 'a_i_OG'] + windowShift  # Change window for sliding
    jointData = jointData.drop(['Start Time', 'End Time', 'Travel Time', 'Mode', 'Dwell Time', 'Trip Distance', 'Destination Reason', 'Urban Size', 'Weight', 'p'], axis=1)
    jointData = jointData.sort_values('Received')


    jointData.loc[:, 'Location0'] = 1
    jointData.loc[:, 'Location1'] = 1

    jointData.loc[jointData.loc[:, 'Type']=='Truck', 'Location0'] = 0

    if(numCars==0):
        jointData = jointData.loc[jointData.loc[:, 'Type']!='Passenger', :]

    if (numTrucks == 0):
        jointData = jointData.loc[jointData.loc[:, 'Type'] != 'Truck', :]

    return jointData

def runFCFS(numSpots, data):
    # Q_FCFS['Vehicle'] = req_master['Vehicle']
    data.loc[:, 'a_i'] = data.loc[:, 'a_i_OG']
    data.loc[:, 'b_i'] = data.loc[:, 'a_i_OG']
    data.loc[:, 's_i'] = data.loc[:, 's_i']
    data.loc[:, 't_i'] = data.loc[:, 'a_i_OG']
    data.loc[:, 'd_i'] = data.loc[:, 'd_i_OG']
    data.loc[:, 'phi'] = data.loc[:, 'phi']
    data.loc[:, 'Prev Assigned'] = 'nan'

    locationCols = getLocationCols(data)
    locationColInds = getLocationColInds(data)
    numSpotsTuple = deepcopy(numSpots)

    data = data.sort_values('a_i')

    parkTimes = []
    for numSpot in numSpots:
        parkTimes.append([-1]*numSpot)
    print(parkTimes)
    assignedCol = []

    startIndex = list(data.columns).index('a_i')
    endIndex = list(data.columns).index('d_i')

    for row in data.itertuples(index=False):
        tempStart = row[startIndex]
        tempEnd = row[endIndex]
        tempAssigned = False
        k = 0
        for locationColInd in locationColInds:
            tempNumSpots = numSpots[k]
            tempAcceptable = row[locationColInd]
            if(tempAcceptable==1 and tempAssigned == False):
                for j in range(tempNumSpots):
                    print(parkTimes[k])
                    tempPrevEnd = parkTimes[k][j]
                    if (not tempAssigned and tempPrevEnd < tempStart):
                        assignedCol.append(1)
                        tempAssigned = True
                        parkTimes[k][j] = tempEnd
            k+=1
        if(not tempAssigned):
            assignedCol.append(0)


    data.loc[:, 'Assigned'] = assignedCol

    return data




    # r = seq_curb(numSpots, data, (6 * 24))


def runFullSetOfResults(numSpots, data, buffer, tau, weightDoubleParking, weightCruising, saveIndex=0):
    r = {}

    buffer+=1

    t0 = datetime.now()

    try:
        r['FCFS'] = runFCFS(numSpots, deepcopy(data))
    except Exception as e:
        r['FCFS'] = e

    t1 = datetime.now()
    fullData = deepcopy(data)
    fullTau = max(fullData.loc[:, 'Received'])-min(fullData.loc[:, 'Received'])-1
    fullTau = 1000
    fullData.loc[:, 'Received_OG'] = fullData.loc[:, 'Received']
    fullData.loc[:, 'Received'] = -1

    # try:
    #     r['full'] = runFullOptimization(deepcopy(data), numSpots, buffer, weightDoubleParking, weightCruising)['data']
    # except:
    #     r['full'] = None
    try:
        r['full'] = runSlidingOptimization(fullData, numSpots, tau=fullTau, buffer=buffer,
                                           weightDoubleParking=weightDoubleParking, weightCruising=weightCruising,
                                           timeLimit=10 * 60)
    except Exception as e:
        r['full'] = e
    t2 = datetime.now()
    try:
        r['sliding'] = runSlidingOptimization(deepcopy(data), numSpots, tau=tau, buffer=buffer,
                                              weightDoubleParking=weightDoubleParking, weightCruising=weightCruising, timeLimit=tau*60)
    except Exception as e:
        r['sliding'] = e
    t3 = datetime.now()
    r['spec'] = {'numSpots':numSpots, 'buffer':buffer, 'tau':tau, 'weightDoubleParking':weightDoubleParking, 'weightCruising':weightCruising}

    r['FCFS-time'] = t1-t0
    r['full-time'] = t2 - t1
    r['sliding-time'] = t3 - t2

    try:
        r['FCFS-unassigned'] = getNumUnassignedMinutes(r['FCFS'])
    except Exception as e:
        r['FCFS-unassigned'] = e

    try:
        r['sliding-unassigned'] = getNumUnassignedMinutes(r['sliding'])
    except Exception as e:
        r['sliding-unassigned'] = e

    try:
        r['full-unassigned'] = getNumUnassignedMinutes(r['full'])
    except Exception as e:
        r['full-unassigned'] = e


    saveFile = '/Users/connorforsythe/Library/CloudStorage/Box-Box/CMU/SmartCurbs/Results/2023-9-12-Prelim/Res-{}.dat'.format(saveIndex)

    with open(saveFile, 'wb') as file:
        pickle.dump(r, file)
        file.close()



    return r

def runFullSetOfResultsQuick(numSpots, data, buffer, tau, weightDoubleParking, weightCruising, saveIndex=0):
    r = {}

    buffer += 1

    t0 = datetime.now()

    try:
        r['FCFS'] = runFCFS(numSpots, deepcopy(data))
    except Exception as e:
        r['FCFS'] = e

    t1 = datetime.now()
    try:
        r['sliding'] = runSlidingOptimization(deepcopy(data), numSpots, tau=tau, buffer=buffer,
                                              weightDoubleParking=weightDoubleParking, weightCruising=weightCruising,
                                              timeLimit=1 * 60)
    except Exception as e:
        r['sliding'] = e
    t2 = datetime.now()

    r['sliding-time'] = t2 - t1
    fullData = deepcopy(data)
    fullTau = max(fullData.loc[:, 'Received']) - min(fullData.loc[:, 'Received']) - 1
    fullTau = 20000
    fullData.loc[:, 'Received_OG'] = fullData.loc[:, 'Received']
    fullData.loc[:, 'Received'] = -1

    try:
        r['full'] = runSlidingOptimization(fullData, numSpots, tau=fullTau, buffer=buffer,
                                           weightDoubleParking=weightDoubleParking, weightCruising=weightCruising,
                                           timeLimit=min(r['sliding-time'].seconds+1, 1*60))
    except Exception as e:
        r['full'] = e
    t3 = datetime.now()
    r['spec'] = {'numSpots': numSpots, 'buffer': buffer, 'tau': tau, 'weightDoubleParking': weightDoubleParking,
                 'weightCruising': weightCruising}




    r['FCFS-time'] = t1 - t0

    r['full-time'] = t3 - t2

    try:
        r['FCFS-unassigned'] = getNumUnassignedMinutes(r['FCFS'])
    except Exception as e:
        r['FCFS-unassigned'] = e

    try:
        r['sliding-unassigned'] = getNumUnassignedMinutes(r['sliding'])
    except Exception as e:
        r['sliding-unassigned'] = e

    try:
        r['full-unassigned'] = getNumUnassignedMinutes(r['full'])
    except Exception as e:
        r['full-unassigned'] = e

    saveFile = '/Users/connorforsythe/Library/CloudStorage/Box-Box/CMU/SmartCurbs/Results/2023-7-31-1MinLimit/Res-{}.dat'.format(
        saveIndex)

    with open(saveFile, 'wb') as file:
        pickle.dump(r, file)
        file.close()

    return r

def getNumUnassignedMinutes(data):
    return np.sum(data.loc[data.loc[:, 'Assigned']==0, 's_i'])

if __name__=='__main__':
    pass
    # #Data construction
    np.random.seed(8131970)
    np.random.seed(11111)
    # np.random.seed(111)
    a = load_nhts_data()
    j = simulateData(1,20, a)

    # r = runSlidingOptimization(j, (1,2), buffer=1, tau = 5)

    r = runFCFS((1,1), j)

    # # j.loc[:, 'Received'] = min(j.loc[:, 'Received'])
    # jc = deepcopy(j)
    # # j.loc[1, 's_i'] = 1
    # r = runFullSetOfResults(50, j, 0, 10, 100, 0, 25)
    # r['FCFS'].loc[:, 'Not Assigned'] = 1-r['FCFS'].loc[:, 'Assigned']
    # r['sliding'].loc[:, 'Not Assigned'] = 1 - r['sliding'].loc[:, 'Assigned']
    # r['full'].loc[:, 'Not Assigned'] = 1 - r['full'].loc[:, 'Assigned']
    # r['FCFS'] = r['FCFS'].sort_values(['Not Assigned', 'a_i_OG'])
    # r['sliding'] = r['sliding'].sort_values(['Not Assigned', 'a_i_Final'])
    # r['full'] = r['full'].sort_values(['Not Assigned', 'a_i_Final'])
    # f = r['FCFS']
    # s = r['sliding']
    # o = r['full']
    #
    # print('FCFS: {}'.format(getNumUnassignedMinutes(f)))
    # print('Sliding: {}'.format(getNumUnassignedMinutes(s)))
    # print('Full: {}'.format(getNumUnassignedMinutes(o)))

    # r2 = runFullSetOfResults(3, j, 10, 10, 1, 99)

    #
    # t0 = datetime.now()
    # r = runSlidingOptimization(jointData, 1, buffer=15) #Change number of parking spots and buffer
    # t1 = datetime.now()
    # print('Running full optimization...')
    # preRFull = runFullOptimization(jointData, 1, buffer=15) #Change number of parking spots and buffergurobi
    # print('Full optimization run...')
    # t2 = datetime.now()
    #
    # slideTime = t1-t0
    # fullTime = t2-t1
    #
    # print('Sliding Time - {}'.format(slideTime.seconds))
    # print('Full Time - {}'.format(fullTime.seconds))
    #
    # x = list(range(10))
    # y = list(range(10))
    #
    # args = []
    # for i in range(10):
    #     args.append((x,y))
    #
    # f = lambda x: x**4
    #
    # with mp.Pool(processes=None) as pool:
    #     r = pool.starmap(f, x)





