import pandas as p
import numpy as np
from SimulateNHTSData import load_nhts_data, select_n_trips, construct_truth_dataframe, gen_truck_arrivals, join_requests
from copy import deepcopy
import gurobipy as gp
from gurobipy import GRB
from collections import OrderedDict
from tqdm import tqdm, trange
from datetime import datetime
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

#This function creates a constraint that limits the number of parking spaces considered
def createNumberParkingSpotConstraint(model, x_i_j, num_spaces=1):
    model.addConstr(x_i_j.sum(0, '*') <= num_spaces, name='NumSpaces')
    return model, x_i_j

#This function creates a set of constraints that preserve flow across X matrix decision variable
def createFlowPreservationConstraints(model, x_i_j, data):
    for i in range(len(data)+1):
        model.addConstr((x_i_j.sum(i, '*') - x_i_j.sum('*', i)) == 0, name='FlowPres-{}'.format(i))

    model.update()
    return model, x_i_j

#Applies constraint requiring that each vehicle is only assigned to a parking spot once
def createSingleAssignmentConstraints(model, x_i_j, data):
    for i in range(1, len(data)+1):
        model.addConstr((x_i_j.sum(i, '*')) <= 1, name='SingleAssignment-{}'.format(i))

    model.update()
    return model, x_i_j


#This function generates and return the Big M matrix for access throughout all other functions
def getBigMMatrix(data, start_scenario=0, end_scenario=(60*24), buffer=5):

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

    for i in range(len(data)):

        # model.addConstr(((t_i[i] + data['s_i'].iloc[i] + buffer) / scale) <= (end_scenario / scale), name='ParkPast-{}'.format(i)) #TODO This constraint is causing infeasible solutions and really isn't necessary
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
            if(j==0):
                #Previous decisions with flow heading back to the "depot" need not be respected as
                #The flow could move to a newly-added vehicle
                model.addConstr(x_i_j[i,j] <= prev_x_i_j[i,j], name='Prev-X-{}-{}'.format(i,j))
            else:
                model.addConstr(x_i_j[i, j] == prev_x_i_j[i, j], name='Prev-X-{}-{}'.format(i, j))
            x_i_j[i,j].Start = prev_x_i_j[i, j]

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


#Returns the objective function associated with double-parking
def setModelObjective(model, *args):
    for obj in args:
        model.setObjective(obj, GRB.MINIMIZE)

    model.update()
    return model

#Returns a pretty version of the optimization output
def constructOutputs(model, t_i, x_i_j, data):
    # Outputs of Interest
    # t = model.status, model.getObjective().getValue()

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

    r = {'t': end_state_t_i, 'x': end_state_x_ij, 'data':data}

    return r

def applyAssignedTimesToData(returnDict):
    cleanOutputs = constructOutputs(*list(returnDict.values()))
    cleanOutputs['data'].loc[:, 'a_i_Final'] = cleanOutputs['t']
    cleanOutputs['data'].loc[:, 'd_i_Final'] = cleanOutputs['data'].loc[:, 'a_i_Final']+cleanOutputs['data'].loc[:, 's_i']
    cleanOutputs['data'].loc[:, 'Assigned'] = np.sum(cleanOutputs['x'], axis=1)[1:]
    return cleanOutputs

def getUnchangingIndicies(outcomeList, currentTime):
    # print('*'*100)
    # print(outcomeList[-1])
    # print(outcomeList[-2])

    #Take out data for easy access
    data0 = outcomeList[-2]['data']
    data1 = outcomeList[-1]['data']

    # Collect shared indices
    index0 = set(list(data0.index))
    index1 = set(list(data1.index))

    sharedIndices = index0 & index1

    if(len(sharedIndices)==0):
        return None

    print(index0)
    print(index1)
    print(sharedIndices)
    #Store binary identification of shared indicies across data
    sharedIndices0 = data0.index.isin(sharedIndices)
    sharedIndices1 = data1.index.isin(sharedIndices)

    #Create separate set of shared indices for x, since x also includes the depot
    sharedXIndices0 = [True]
    sharedXIndices1 = [True]

    sharedXIndices0.extend(sharedIndices0)
    sharedXIndices1.extend(sharedIndices1)

    print(sharedIndices0)
    print(sharedIndices1)


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

    # print('Vehicle 0: {}'.format(list(outcomeList[-2]['data'].loc[:, 'Vehicle'])))
    # print('Vehicle 1: {}'.format(list(outcomeList[-1]['data'].loc[:, 'Vehicle'])))
    # print('x0: {}'.format(x0))
    # print('x1: {}'.format(x1))
    # print('xDiff: {}'.format(xDiff))
    # print('t0: {}'.format(t0))
    # print('t1: {}'.format(t1))
    # print('tDiff: {}'.format(tDiff))


    #Indices that are finished parking
    preIndices = data1.loc[sharedIndices, 'd_i_Final']<currentTime
    print('Pre-Indices: {}'.format(preIndices))
    #If things remain the same between time steps and a given vehicle is no
    #longer relevant they can be dropped. Inverse is also stored.
    indicesToDrop = (tDiff==0)&(preIndices)&(xDiff[1:, :].sum(axis=1)==0)
    indicesToKeep = np.logical_not(indicesToDrop)

    xIndicesToDrop = [True]
    xIndicesToKeep = [True]

    xIndicesToDrop.extend(list(indicesToDrop))
    xIndicesToKeep.extend(list(indicesToKeep))

    #&(xDiff[1:, :].sum(axis=1)==0)

    # print('Indices to Drop: {}'.format(indicesToDrop))
    # print('Indices to Keep: {}'.format(indicesToKeep))

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
    print(r['unlocked']['x'])

    r['unlocked']['x'] = r['unlocked']['x'][xUnlockedIndices, :]
    r['unlocked']['x'] = r['unlocked']['x'][:, xUnlockedIndices]
    if(len(r['locked']['t']>0)):
        print(np.array(xUnlockedIndices).shape)
        print(outcomeList[-1]['x'].shape)
        print(np.array(xUnlockedIndices).shape)
        print(xIndicesToDrop)
        print(r['unlocked']['data'].loc[:, 'Assigned'].shape)
        r['unlocked']['x'][0, :] = outcomeList[-1]['x'][:, xUnlockedIndices].sum(axis=0)-np.multiply(r['unlocked']['x'][1:, :].sum(axis=0), 1)
        r['unlocked']['x'][0, 0] = 0




    # print('Locked: {}'.format(r['locked']))
    # print('Unlocked: {}'.format(r['unlocked']))
    # print('Indices 0: {}'.format(list(data0.index)))
    # print('Indices 1: {}'.format(list(data1.index)))
    # print('*' * 100)
    return r


#Runs the full optimization routine
def runFullOptimization(data, numSpaces):
    # Model construction and optimization
    m = createGurobiModel()
    m, t_i = createTimeVars(m, data)
    m, x_i_j = createFlowVars(m, data)
    m, x_i_j = createNumberParkingSpotConstraint(m, x_i_j, num_spaces=numSpaces)
    m, x_i_j = createFlowPreservationConstraints(m, x_i_j, data)
    m, x_i_j = createSingleAssignmentConstraints(m, x_i_j, data)
    bigM = getBigMMatrix(data)
    m, t_i, x_i_j = createTimeOverlapConstraints(m, t_i, x_i_j, data)
    m, t_i = createTimeWindowConstraints(m, t_i, data) #TODO HERE IS THE ISSUE
    m, t_i, x_i_j = createTimeShiftConstraints(m, t_i, x_i_j, data)
    m = setModelParams(m, MIPGap=0.01)
    doubleParkObj = getDoubleParkExpression(data, x_i_j)
    m = setModelObjective(m, doubleParkObj)
    t0 = m.status
    m.optimize()
    m.update()
    print('Termination Code: {}'.format(m.status))

    if(m.status==2):
        r = OrderedDict({'model':m, 't':t_i, 'x':x_i_j, 'data':data})
    else:
        r = m.status


    return r

def runSlidingOptimization(data, numSpaces, tau=5, start=0, stop=(24*60)+1):
    # Model construction and optimization
    #r is a list that will store successive results over time
    #Data should be sorted by received so that constraints moving from one time
    #step to the next are not assigned to the wrong vehicles
    data = data.sort_values('Received')
    data.index = range(len(data))
    prevLen = 0
    r = []
    indiciesToDrop = []
    iterVar = trange(start, stop, tau)
    print('Iter: {}'.format(range(start, stop, tau)))
    passPrevDecisions = False
    for j in iterVar:
        data = data.loc[~data.index.isin(indiciesToDrop), :]
        tempData = data.loc[data.loc[:, 'Received']<=j, :]

        if(len(tempData)>0 and len(tempData)>prevLen): #
            m = createGurobiModel()
            m, t_i = createTimeVars(m, tempData)
            m, x_i_j = createFlowVars(m, tempData)
            m, x_i_j = createNumberParkingSpotConstraint(m, x_i_j, num_spaces=numSpaces)
            m, x_i_j = createFlowPreservationConstraints(m, x_i_j, tempData)
            m, x_i_j = createSingleAssignmentConstraints(m, x_i_j, tempData)
            m, t_i, x_i_j = createTimeOverlapConstraints(m, t_i, x_i_j, tempData)
            m, t_i = createTimeWindowConstraints(m, t_i, tempData)
            m, t_i, x_i_j = createTimeShiftConstraints(m, t_i, x_i_j, tempData)
            m = setModelParams(m, TimeLimit=45, MIPGap=0.01)
            doubleParkObj = getDoubleParkExpression(tempData, x_i_j)
            m = setModelObjective(m, doubleParkObj)

            # Drop vehicle considered whenever answers haven't changed
            if (len(r) > 1):
                unchangingDict = getUnchangingIndicies(r, j)

                # if(len(unchangingDict['unlocked']['t']>0)):
                #
                #     m, t_i = createPersistentTimeDecisionConstraints(m, t_i, r[-1]['data'])
                #     m, x_i_j = createPersistentFlowDecisionConstraints(m, x_i_j, r[-1]['x'])
                #
                #     passPrevDecisions = False
                # else:
                #     passPrevDecisions = False


                if(True):
                    x0 = r[-2]['x']
                    x1 = r[-1]['x']
                    if (len(x0) <= len(x1)):
                        x0Len = len(x0)
                        diff = x1[:len(x0), :len(x0)] - x0
                        diffValue = np.sum(np.abs(diff))
                    else:
                        diffValue = 1

                    if (diffValue == 0):
                        indiciesToDrop.extend(list(r[-2]['data'].index))
                        # indiciesToDrop = list(set(indiciesToDrop))
                        passPrevDecisions = False
                    else:
                        passPrevDecisions = True
            else:
                passPrevDecisions = True
            t0 = m.status
            m.optimize()
            m.update()
            print('Termination Code: {}'.format(m.status))

            #Now set constraints on previous decisions should there be any
            if(len(r)>0):
                m, t_i = createPersistentTimeDecisionConstraints(m, t_i, r[-1]['data'])
                m, x_i_j = createPersistentFlowDecisionConstraints(m, x_i_j, r[-1]['x'])



            if(m.status==2):
                tempR = OrderedDict({'model':m, 't':t_i, 'x':x_i_j, 'data':tempData})
                tempR = applyAssignedTimesToData(tempR)
            else:
                tempR = m.status
                tempR = OrderedDict({'model': m, 't': t_i, 'x': x_i_j, 'data': tempData})
                tempR = applyAssignedTimesToData(tempR)

            tempR['passPrevDecisions'] = passPrevDecisions
            tempR['indiciesToDrop'] = list(set(indiciesToDrop))
            tempR['indiciesToDrop'] = indiciesToDrop
            tempR['status'] = m.status
            prevLen = len(list(set(indiciesToDrop)))
            r.append(tempR)
    return r

def runSlidingOptimization2(data, numSpaces, tau=5, start=0, stop=(24*60)+1):
    # Model construction and optimization
    #r is a list that will store successive results over time
    #Data should be sorted by received so that constraints moving from one time
    #step to the next are not assigned to the wrong vehicles
    data = data.sort_values('Received')
    data.index = range(len(data))
    prevLen = 0
    r = []
    unchangingDicts = []
    indiciesToDrop = []
    iterVar = trange(start, stop, tau)
    print('Iter: {}'.format(range(start, stop, tau)))
    passPrevDecisions = False
    for j in iterVar:

        if (len(r) > 1):
            unchangingDict = getUnchangingIndicies(r, j)
            print(unchangingDict)
            if(unchangingDict!=None):
                unchangingDicts.append(unchangingDict)
                indiciesToDrop.extend(unchangingDict['locked']['indices'])
        print('Indices to drop: {}'.format(indiciesToDrop))
        data = data.loc[~data.index.isin(indiciesToDrop), :]
        tempData = data.loc[data.loc[:, 'Received'] <= j, :]

        if(len(tempData)>0): #
            #Create Model
            m = createGurobiModel()
            m, t_i = createTimeVars(m, tempData)
            m, x_i_j = createFlowVars(m, tempData)

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
            m, x_i_j = createSingleAssignmentConstraints(m, x_i_j, tempData)
            m, t_i, x_i_j = createTimeOverlapConstraints(m, t_i, x_i_j, tempData)
            m, t_i = createTimeWindowConstraints(m, t_i, tempData)
            m, t_i, x_i_j = createTimeShiftConstraints(m, t_i, x_i_j, tempData)
            m = setModelParams(m, TimeLimit=45, MIPGap=0.01)
            doubleParkObj = getDoubleParkExpression(tempData, x_i_j)
            m = setModelObjective(m, doubleParkObj)



            t0 = m.status
            m.optimize()
            m.update()
            print('Termination Code: {}'.format(m.status))

            if(m.status==2):
                tempR = OrderedDict({'model':m, 't':t_i, 'x':x_i_j, 'data':tempData})
                tempR = applyAssignedTimesToData(tempR)
            else:
                tempR = m.status
                tempR = OrderedDict({'model': m, 't': t_i, 'x': x_i_j, 'data': tempData})
                tempR = applyAssignedTimesToData(tempR)

            tempR['indiciesToDrop'] = list(set(indiciesToDrop))
            tempR['status'] = m.status
            tempR['Current Time'] = j
            r.append(tempR)


    r = collateFinalOutcomes(r)
    r = r.sort_values('Received')
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
        prevOutcome = finalOutcomes[i-1]
        outcome = finalOutcomes[i]

        prevData = prevOutcome['data']
        data = outcome['data']

        prevVehicles = set(prevData.loc[:, 'Vehicle'])
        vehicles = set(data.loc[:, 'Vehicle'])

        lostVehicles = prevVehicles-vehicles

        # count+=len(outcome['data'])
        datas.append(prevData.loc[prevData.loc[:, 'Vehicle'].isin(lostVehicles), :])


    datas.append(outcome['data'])
    datas = p.concat(datas)

    return datas

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

    # print('Count: {}'.format(count))
    return outcomeList, r

if __name__=='__main__':
    #Data construction
    np.random.seed(8131970)
    data = load_nhts_data()
    r = OrderedDict()

    t_sample = select_n_trips(data, num_sample=25)
    t = construct_truth_dataframe(t_sample)

    truck = gen_truck_arrivals(1)

    jointData = join_requests(t, truck)
    jointData.loc[:, 'b_i_OG'] = jointData.loc[:, 'a_i_OG'] + 10
    jointData = jointData.sort_values('Received')
    # early = jointData.loc[jointData.loc[:, 'Received'] < 700, :]
    # preR = runFullOptimization(early, 2)
    # r = constructOutputs(*list(preR.values()))
    # r = applyAssignedTimesToData(preR)

    t0 = datetime.now()
    r = runSlidingOptimization2(jointData, 1)
    # preR52 = preR[52]
    # preR53 = preR[53]
    # preR65 = preR[65]
    # preR66 = preR[66]
    t1 = datetime.now()
    print('Running full optimization...')
    preRFull = runFullOptimization(jointData, 1)
    print('Full optimization run...')
    t2 = datetime.now()

    slideTime = t1-t0
    fullTime = t2-t1

    # finalOutcomes = collateFinalOutcomes(preR)

    print('Sliding Time - {}'.format(slideTime.seconds))
    print('Full Time - {}'.format(fullTime.seconds))

    # veh = []
    # indices = []
    # for t in preR:
    #     veh.extend(list(t['data'].loc[:, 'Vehicle']))
    #
    # veh = list(set(veh))
    # dataAddressed = jointData.loc[jointData.loc[:, 'Vehicle'].isin(veh), :]
    # dataUnaddressed = jointData.loc[~jointData.loc[:, 'Vehicle'].isin(veh), :]

