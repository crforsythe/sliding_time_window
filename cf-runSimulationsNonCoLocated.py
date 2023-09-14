from multiprocessing import Pool
import multiprocessing as mp
from cfImplementationNonCoLocated import runFullSetOfResults, simulateData, load_nhts_data
import numpy as np



if __name__ == '__main__':

    # numSpots = list(range(0, 51,10))
    # numSpots[0] = 1
    numSpots = [(0,1), (1,1)]
    totalNumVehicles = list(range(10,31,10))
    # totalNumVehicles = [405]
    doubleParkWeights = [100, 50, 0]
    bufferValues = [0,5]
    # bufferValues = [0]
    tauValues = [5]
    truckProps = [0, .5, 1]
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
            sumSpot = np.sum(numSpot)
            totalNumVehicles = list(range(11*sumSpot, 34*sumSpot, 11*sumSpot))
            # totalNumVehicles = list(range(11, 78, 11))
            for numVehicles in totalNumVehicles:
                # numTrucks = list(range(1*numSpot, numVehicles*numSpot, 10*numSpot))

                for truckProp in truckProps:
                    numTruck = int(np.round(truckProp*numVehicles))
                    numCar = numVehicles-numTruck
                    tempData = simulateData(numCar, numTruck, nhts_data)
                    args = []
                    for doubleParkWeight in doubleParkWeights:
                        cruisingWeight = 100-doubleParkWeight
                        for tauValue in tauValues:
                            for bufferValue in bufferValues:
                                tempArg = (numSpot, tempData, bufferValue, tauValue, doubleParkWeight, cruisingWeight, i)
                                # runFullSetOfResults(*tempArg)
                                if(sumSpot==1 and doubleParkWeight==100 and numVehicles==77):
                                    args.append(tempArg)
                                else:
                                    args.append(tempArg)
                                    pass
                                i+=1

    # # mp.set_start_method('fork')
                    numThreads = mp.cpu_count()-2
                    chunkSize = max(int(len(args)/numThreads), 1)
                    np.random.shuffle(args)
                    with Pool(numThreads) as pool:
                        r = pool.starmap(runFullSetOfResults, args, chunksize=chunkSize)
                        pool.close()