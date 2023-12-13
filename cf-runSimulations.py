from multiprocessing import Pool
import multiprocessing as mp
from cfImplementation import runFullSetOfResults, simulateData, load_nhts_data
import numpy as np
import pickle #added
import glob #added


def gen_vehicles_and_parameters(replications, numSpots, truckProps, nhts_data, 
                                doubleParkWeights, tauValues, zetaValues, saveIndex=1):
    
    args = []
    i = 0
    reps = range(replications)
    for rep in reps:
        for numSpot in numSpots:
            totalNumVehicles = list(range(11*numSpot, 34*numSpot, 11*numSpot))
            # totalNumVehicles = list(range(11, 78, 11))
            for numVehicles in totalNumVehicles:
                # numTrucks = list(range(1*numSpot, numVehicles*numSpot, 10*numSpot))

                for truckProp in truckProps:
                    numTruck = int(np.round(truckProp*numVehicles))
                    numCar = numVehicles-numTruck
                    tempData = simulateData(min(max(numCar, 0), numVehicles), min(max(numTruck, 0), numVehicles), nhts_data)
                    #args = []
                    for doubleParkWeight in doubleParkWeights:
                        cruisingWeight = 100-doubleParkWeight
                        for tauValue in tauValues:
                            for bufferValue in bufferValues:
                                for zetaValue in zetaValues:
                                    #tempArg = (numSpot, tempData, bufferValue, zetaValue, doubleParkWeight, cruisingWeight, i, tauValue)
                                    tempArg = {'numSpot': numSpot, 
                                                   'tempData': tempData, 
                                                   'bufferValue': bufferValue, 
                                                   'zetaValue': zetaValue, 
                                                   'doubleParkWeight': doubleParkWeight, 
                                                   'cruisingWeight': cruisingWeight, 
                                                   'i': i, 
                                                   'tauValue': tauValue}
                                    
                                    if (numSpot == 1 and doubleParkWeight == 100 and numVehicles == 77):
                                        args.append(tempArg)
                                    else:
                                        args.append(tempArg)
                                        pass
                                    i += 1
                                    print(i) 
    
    
    saveFile = 'AaronRes/Veh_and_Params.dat'.format(saveIndex)

    with open(saveFile, 'wb') as file:
        pickle.dump(args, file)
        file.close()
    
    return args



if __name__ == '__main__':

    #numSpots = [1, 2, 5, 10, 25]
    numSpots = [1, 2, 10]
    # numSpots = [1]
    # totalNumVehicles = list(range(11,78,11))
    # totalNumVehicles = [405]
    #doubleParkWeights = range(0, 101,25)
    doubleParkWeights = [100]
    bufferValues = [0]
    # bufferValues = [0]
    tauValues = [5*60, 0]
    #zetaValues = [1,5]
    zetaValues = [5]
    truckProps = [0]
    # tauValues = [5]
    replications = 1 #added by Aaron
    windowShift = 10 #added by Aaron
    rhoValues = [0, 60]
    nuValues = [0, 60]

    # numSpots = [10]
    # numSpots[0] = 1
    # numTrucks = list(range(5, 506,500))
    # numCars = list(500-np.array(numTrucks)+10)
    # doubleParkWeights = range(0, 101,100)
    # cruisingWeights = list(100-np.array(doubleParkWeights))
    # bufferValues = [10]
    # tauValues = [10]



    np.random.seed(3102023)
    np.random.seed(16112023)

    nhts_data = load_nhts_data(windowShift)  #added windowShift
    
    # args = gen_vehicles_and_parameters(replications, numSpots, truckProps, nhts_data, 
    #                                 doubleParkWeights, tauValues, zetaValues)

    # #debugging workflow
    #
    # fp = 'AaronRes/Veh_and_Params.dat'
    # files = glob.glob(fp)
    #
    # args = []
    # for file in files:
    #     with open(file, 'rb') as temp:
    #         args.append(pickle.load(temp))
    #         temp.close()
    # args = args[0]
    #
    # run = args[8]
    # outcomes = runFullSetOfResults(idx = run['i'],
    #                     numSpots = 1,
    #                     data = run['tempData'],
    #                     buffer = run['bufferValue'],
    #                     zeta = run['zetaValue'],
    #                     weightDoubleParking = run['doubleParkWeight'],
    #                     weightCruising = run['cruisingWeight'],
    #                     tau = run['tauValue'])
    #
    #
    # slidingSorted = outcomes['sliding'].sort_values(['Assigned', 'a_i_Final'], ascending=[False, True])
    # fullSorted = outcomes['full'].sort_values(['Assigned', 'a_i_Final'], ascending=[False, True])
    # fcfsSorted = outcomes['FCFS'].sort_values(['Assigned', 'a_i_OG'], ascending=[False, True])

    # slidingFull = outcomes['sliding'][1]
    # slidingPastInfo = outcomes['sliding'][2]
    #run for record workflow
    # nhts_data = load_nhts_data(windowShift) 
    
    # args = gen_vehicles_and_parameters(replications, numSpots, truckProps, nhts_data, 
    #                                 doubleParkWeights, tauValues, zetaValues)
    
    # arg_tuples = []
    # for arg_set in args:
    #     arg_tuple = (arg_set['i'], arg_set['numSpot'], arg_set['tempData'], arg_set['bufferValue'],
    #                   arg_set['zetaValue'], arg_set['doubleParkWeight'], arg_set['cruisingWeight'],
    #                   0, arg_set['tauValue']) #0 is for the saveIndex input parameter
    #     arg_tuples.append(arg_tuple)
    
    # numThreads = mp.cpu_count()-2
    # #numThreads = 4
    # chunkSize = max(int(len(arg_tuples)/numThreads), 1)
    # np.random.shuffle(arg_tuples)
    # with Pool(numThreads) as pool:
    #     r = pool.starmap(runFullSetOfResults, arg_tuples, chunksize=chunkSize)
    #     pool.close()
    
    
    
    
    
# Connor's original code   

    i = 0
    reps = range(0,3)
    for rep in reps:
        for numSpot in numSpots:
            totalNumVehicles = list(range(11*numSpot, 34*numSpot, 11*numSpot))
            # totalNumVehicles = list(range(11, 78, 11))
            for numVehicles in totalNumVehicles:
                # numTrucks = list(range(1*numSpot, numVehicles*numSpot, 10*numSpot))

                for truckProp in truckProps:
                    numTruck = int(np.round(truckProp*numVehicles))
                    numCar = numVehicles-numTruck
                    tempData = simulateData(min(max(numCar, 0), numVehicles), min(max(numTruck, 0), numVehicles), nhts_data)
                    args = []
                    for doubleParkWeight in doubleParkWeights:
                        cruisingWeight = 100-doubleParkWeight
                        for tauValue in tauValues:
                            for bufferValue in bufferValues:
                                for zetaValue in zetaValues:
                                    for rhoValue in rhoValues:
                                        for nuValue in nuValues:
                                            tempArg = (numSpot, tempData, bufferValue, zetaValue, doubleParkWeight, cruisingWeight, i, tauValue, rhoValue, nuValue) #, rhoValue, nuValue
        
                                            # if i == 8:
                                            #     print('i = ' + str(i))
                                            runFullSetOfResults(*tempArg)
                                            if (numSpot == 1 and doubleParkWeight == 100 and numVehicles == 77):
                                                args.append(tempArg)
                                            else:
                                                args.append(tempArg)
                                                pass
                                            i += 1
                                            print(i)
        
                    #mp.set_start_method('fork')
                    numThreads = mp.cpu_count()-2
                    #numThreads = 4
                    chunkSize = max(int(len(args)/numThreads), 1)
                    np.random.shuffle(args)
                    with Pool(numThreads) as pool:
                        r = pool.starmap(runFullSetOfResults, args, chunksize=chunkSize)
                        pool.close()