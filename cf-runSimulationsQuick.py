from multiprocessing import Pool
import multiprocessing as mp
from cfImplementation import runFullSetOfResults, simulateData, load_nhts_data, runFullSetOfResultsQuick
import numpy as np

def alterReceived(data, newReceivedDelta=5):
    data.loc[:, 'Received-Diff'] = data.loc[:, 'a_i_OG']-data.loc[:, 'Received']
    greaterThanInds = data.loc[:, 'Received-Diff']>newReceivedDelta
    data.loc[greaterThanInds, 'Received'] = data.loc[greaterThanInds, 'a_i_OG']-newReceivedDelta
    return data


if __name__ == '__main__':

    numSpots = list(range(0, 51,10))
    numSpots[0] = 1
    # numSpots = [1]
    totalNumVehicles = list(range(11,78,11))
    # totalNumVehicles = [10]
    doubleParkWeights = range(0, 101,25)
    # doubleParkWeights = [100]
    bufferValues = [0,5]
    # bufferValues = [0]
    tauValues = [2,5]
    truckProps = [0, .25, .5, .75, 1]
    receivedDeltas = [30, 15, 5]
    # tauValues = [5]

    # numSpots = [10]
    # numSpots[0] = 1
    # numTrucks = list(range(5, 506,500))
    # numCars = list(500-np.array(numTrucks)+10)
    # doubleParkWeights = range(0, 101,100)
    # cruisingWeights = list(100-np.array(doubleParkWeights))
    # bufferValues = [10]
    # tauValues = [10]



    np.random.seed(3102023)

    nhts_data = load_nhts_data()

    i = 0
    reps = range(5)
    for rep in reps:

        for numSpot in numSpots:
            totalNumVehicles = list(range(11*numSpot, 34*numSpot, 11*numSpot))
            # totalNumVehicles = list(range(11, 78, 11))
            for numVehicles in totalNumVehicles:
                # numTrucks = list(range(1*numSpot, numVehicles*numSpot, 10*numSpot))

                for truckProp in truckProps:
                    numTruck = int(np.round(truckProp*numVehicles))
                    numCar = numVehicles-numTruck
                    tempData = simulateData(min(max(numCar, 1), numVehicles - 1),
                                            min(max(numTruck, 1), numVehicles - 1), nhts_data)
                    for receivedDelta in receivedDeltas:
                        tempData = alterReceived(tempData, receivedDelta)
                        args = []
                        for doubleParkWeight in doubleParkWeights:
                            cruisingWeight = 100-doubleParkWeight
                            for tauValue in tauValues:
                                for bufferValue in bufferValues:
                                    print('{}-{}'.format(i, receivedDelta))
                                    tempArg = (numSpot, tempData, bufferValue, tauValue, doubleParkWeight, cruisingWeight, i)
                                    # runFullSetOfResults(*tempArg)
                                    if(numSpot==1 and doubleParkWeight==100 and numVehicles==77):
                                        args.append(tempArg)
                                    else:
                                        args.append(tempArg)
                                        pass
                                    i+=1

    # # mp.set_start_method('fork')
                        numThreads = mp.cpu_count()-2
                        numThreads = 10
                        chunkSize = max(int(len(args)/numThreads), 1)
                        np.random.shuffle(args)
                        with Pool(numThreads) as pool:
                            r = pool.starmap(runFullSetOfResultsQuick, args, chunksize=chunkSize)
                            pool.close()