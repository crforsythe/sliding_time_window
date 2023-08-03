import pickle
import glob
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

def loadObject(fileName):
    with open(fileName, 'rb') as file:
        r = pickle.load(file)
        r['file'] = fileName
        file.close()

    return r

def getFiles():
    direc = '/Users/connorforsythe/Library/CloudStorage/Box-Box/CMU/SmartCurbs/Results/InitResultsTimeLimited/*.dat'
    direc = '/Users/connorforsythe/Library/CloudStorage/Box-Box/CMU/SmartCurbs/Results/Results-10MinFullOptimization/*.dat'
    files = glob.glob(direc)
    return files

def loadAllObjects():
    files = getFiles()

    r = []

    for file in tqdm(files):
        r.append(loadObject(file))

    r, small = cleanData(r)

    return r, small

def cleanData(data):
    small = []
    for res in data:
        res['FCFS-UnassignedMinutes'] = np.sum((1-res['FCFS'].loc[:, 'Assigned'])*res['FCFS'].loc[:, 's_i'])
        res['sliding-UnassignedMinutes'] = np.sum((1 - res['sliding'].loc[:, 'Assigned']) * res['sliding'].loc[:, 's_i'])
        # try:
        #     res['full-UnassignedMinutes'] = np.sum((1 - res['full'].loc[:, 'Assigned']) * res['full'].loc[:, 's_i'])
        # except:
        #     res['full-UnassignedMinutes'] = np.nan
        res['sliding-FCFS-UnassignedMinutes'] = res['sliding-UnassignedMinutes']-res['FCFS-UnassignedMinutes']
        # res['full-FCFS-UnassignedMinutes'] = res['full-UnassignedMinutes'] - res['FCFS-UnassignedMinutes']
        # res['full-sliding-UnassignedMinutes'] = res['full-UnassignedMinutes'] - res['sliding-UnassignedMinutes']
        res['total-min-sliding'] = np.sum(res['sliding-FCFS-UnassignedMinutes'])
        # res['total-min-full'] = np.sum(res['full-FCFS-UnassignedMinutes'])
        # res['total-min-full-sliding'] = np.sum(res['full-sliding-UnassignedMinutes'])
        small.append(res['total-min-sliding'])
        # res['-UnassignedMinutes'] = (1 - res['sliding'].loc[:, 'Assigned']) * res['sliding'].loc[:, 's_i']

    return data, small

a, small = loadAllObjects()

plt.hist(small)
plt.show()

weird = []
newSmall = []
for res in a:
    newSmall.append(res['sliding-UnassignedMinutes']-res['FCFS-UnassignedMinutes'])
    if(res['sliding-UnassignedMinutes']>res['FCFS-UnassignedMinutes'] and res['spec']['weightDoubleParking']==100 and res['spec']['numSpots']==1):
        for temp in ['FCFS', 'sliding']:
            res[temp].loc[:, 'Not Assigned'] = 1 - res[temp].loc[:, 'Assigned']
            res[temp] = res[temp].sort_values(['Not Assigned', 'a_i_OG'])
        weird.append(res)
        print(res['spec'])