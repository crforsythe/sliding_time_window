# -*- coding: utf-8 -*-
"""
Created on Wed Jul  5 11:14:19 2023

@author: Aaron
"""

import pickle
import glob
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import datetime

# files = glob.glob('C:/Users/Aaron/Box/Results/InitResultsSmall/*.dat')
files = glob.glob(
    '/Users/connorforsythe/Library/CloudStorage/Box-Box/CMU/SmartCurbs/Results/Results-10MinFullOptimization/*.dat')  # much faster option, download the files first

res = []

for file in files:
    with open(file, 'rb') as temp:
        res.append(pickle.load(temp))
        temp.close()

timeFCFS = []
timeFull = []
timeMPC = []
numSpaces = []
numVehicles = []
tau = []
sumSI = []
propCars = []
schedFCFS = []
schedFull = []
schedMPC = []

for run in res:
    timeFCFS.append(run['FCFS-time'].total_seconds())
    timeFull.append(run['full-time'].total_seconds())
    timeMPC.append(run['sliding-time'].total_seconds())
    numSpaces.append(run['spec']['numSpots'])
    numVehicles.append(len(run['FCFS']))
    tau.append(run['spec']['tau'])
    sumSI.append(sum(run['FCFS']['s_i']))
    schedFCFS.append(run['FCFS'])
    # schedFull.append(run['full'])
    schedMPC.append(run['sliding'])

raw_data_df = pd.DataFrame(list(zip(numVehicles, numSpaces, tau, timeFCFS, timeFull, timeMPC)),
                           columns=['Vehicles', 'Parking Spaces', 'tau',
                                    'Runtime FCFS', 'Runtime Full', 'Runtime MPC'])

raw_data_df['demand'] = raw_data_df['Vehicles'] / raw_data_df['Parking Spaces']
raw_data_df = raw_data_df.reindex(columns=['Vehicles', 'Parking Spaces', 'demand', 'tau',
                                           'Runtime FCFS', 'Runtime Full', 'Runtime MPC'])

sns.lineplot(x='demand', y='Runtime FCFS', hue='tau',
             data=raw_data_df, palette='tab10')

# set up the dataframe for seaborn
dataFCFS = pd.DataFrame(list(zip(numVehicles, numSpaces, tau, timeFCFS)),
                        columns=['Vehicles', 'Parking Spaces', 'tau',
                                 'Runtime'])
dataFCFS['demand'] = dataFCFS['Vehicles'] / dataFCFS['Parking Spaces']
dataFCFS['Method'] = 'FCFS'

dataFull = pd.DataFrame(list(zip(numVehicles, numSpaces, tau, timeFull)),
                        columns=['Vehicles', 'Parking Spaces', 'tau',
                                 'Runtime'])
dataFull['demand'] = dataFull['Vehicles'] / dataFull['Parking Spaces']
dataFull['Method'] = 'Full'

dataMPC = pd.DataFrame(list(zip(numVehicles, numSpaces, tau, timeMPC)),
                       columns=['Vehicles', 'Parking Spaces', 'tau',
                                'Runtime'])
dataMPC['demand'] = dataMPC['Vehicles'] / dataMPC['Parking Spaces']
dataMPC['Method'] = 'MPC'

data_df = pd.concat([dataFCFS, dataFull, dataMPC])

data_df.reset_index(inplace=True)

# seaborn plot for runtime
plt.figure()
sns.lineplot(x='demand', y='Runtime', hue='tau', style='Method',
             data=data_df, palette='tab10', ci=None)
plt.xlabel('Parking Space Demand (Num Vehicles / Num Parking Spaces)')
plt.ylabel('Runtime (seconds)')
plt.title('Optimization Runtime given Different Methods and tau Values')
# plt.ylim([-1, 20])
plt.xlim([-1, 25])

plt.figure()
sns.lineplot(x='Vehicles', y='Runtime', hue='tau', style='Method',
             data=data_df, palette='tab10', ci=None)
plt.xlabel('Number of Vehicles)')
plt.ylabel('Runtime (seconds)')
plt.title('Optimization Runtime given Different Methods and tau Values')
# plt.ylim([-1, 20])
# plt.xlim([-1, 25])


plt.figure()
sns.lineplot(x='Parking Spaces', y='Runtime', hue='tau', style='Method',
             data=data_df, palette='tab10', ci=None)
plt.xlabel('Number of Parking Spaces)')
plt.ylabel('Runtime (seconds)')
plt.title('Optimization Runtime given Different Methods and tau Values')
# plt.ylim([-1, 20])
# plt.xlim([-1, 25])

# ------------------------------------------------------------------------------
# Reduction Graphics

totalDblParkFCFS = []
totalCruiseFCFS = []
totalDblParkFull = []
totalCruiseFull = []
totalDblParkMPC = []
totalCruiseMPC = []
numTrucks = []

for ls in schedFCFS:
    ls.replace('Double-Park', 'Double Park',
               inplace=True)  # both entries were present initially, set all to the same string

    dblParkSchedFCFS = ls.loc[(ls['Assigned'] == 0) &
                              (ls['No-Park Outcome'] == 'Double Park')]
    totalDblParkFCFS.append(sum(dblParkSchedFCFS['s_i']))

    cruiseSchedFCFS = ls.loc[(ls['Assigned'] == 0) &
                             (ls['No-Park Outcome'] == 'Cruising')]
    totalCruiseFCFS.append(sum(cruiseSchedFCFS['Expected Cruising Time']))

    numTrucks.append(sum(ls['Type'] == 'Truck'))

for ls in schedFull:
    if ls is not None:  # some weird data issue where 4 of the dataframe were not recorded
        ls.replace('Double-Park', 'Double Park',
                   inplace=True)  # both entries were present initially, set all to the same string

        dblParkSchedFull = ls.loc[(ls['Assigned'] == 0) &
                                  (ls['No-Park Outcome'] == 'Double Park')]
        totalDblParkFull.append(sum(dblParkSchedFull['s_i']))

        cruiseSchedFull = ls.loc[(ls['Assigned'] == 0) &
                                 (ls['No-Park Outcome'] == 'Cruising')]
        totalCruiseFull.append(sum(cruiseSchedFull['Expected Cruising Time']))

for ls in schedMPC:
    ls.replace('Double-Park', 'Double Park',
               inplace=True)  # both entries were present initially, set all to the same string

    dblParkSchedMPC = ls.loc[(ls['Assigned'] == 0) &
                             (ls['No-Park Outcome'] == 'Double Park')]
    totalDblParkMPC.append(sum(dblParkSchedMPC['s_i']))

    cruiseSchedMPC = ls.loc[(ls['Assigned'] == 0) &
                            (ls['No-Park Outcome'] == 'Cruising')]
    totalCruiseMPC.append(sum(cruiseSchedMPC['Expected Cruising Time']))

# calc the reduction metrics

reduxDblParkFCFStoMPC = np.subtract(np.array(totalDblParkFCFS), np.array(totalDblParkMPC))
reduxCruiseFCFStoMPC = np.subtract(np.array(totalCruiseFCFS), np.array(totalCruiseMPC))
# reduxFulltoMPC = np.subtract(np.array(totalDblParkFCFS), np.array(totalDblParkFull))


data_df = pd.DataFrame(
    list(zip(numVehicles, numTrucks, numSpaces, tau, sumSI, reduxDblParkFCFStoMPC, reduxCruiseFCFStoMPC)),
    columns=['Vehicles', 'Trucks', 'Parking Spaces', 'tau', 'sumSI', 'Redux Dbl Park', 'Redux Cruising'])

data_df['Norm Parking Spaces'] = data_df['sumSI'] / data_df['Parking Spaces']
data_df['Prop Trucks'] = data_df['Trucks'] / data_df['Vehicles']

fig = plt.figure()
ax = plt.tricontour(data_df['Norm Parking Spaces'], data_df['Prop Trucks'], data_df['Redux Dbl Park'], cmap='tab10')
cb = fig.colorbar(ax)
plt.title('Reduction in Double Parking (min) between FCFS and MPC')
plt.xlabel('Service Duration Requested per Parking Space')
plt.ylabel('Proportion of Trucks')
plt.ylim([-.05, 1.05])
plt.legend()

fig = plt.figure()
ax = plt.tricontour(data_df['Norm Parking Spaces'], data_df['Prop Trucks'], data_df['Redux Cruising'], cmap='tab10')
cb = fig.colorbar(ax)
plt.title('Reduction in Cruising (min) between FCFS and MPC')
plt.xlabel('Service Duration Requested per Parking Space')
plt.ylabel('Proportion of Trucks')
plt.ylim([-.05, 1.05])
plt.xlim([0, 1300])
plt.legend()

fig = plt.figure()
ax = plt.tricontour(data_df['Vehicles'], data_df['Prop Trucks'], data_df['Redux Dbl Park'], cmap='tab10')
cb = fig.colorbar(ax)
plt.title('Reduction in Dbl Parking (min) between FCFS and MPC')
plt.xlabel('Number of Vehicles')
plt.ylabel('Proportion of Trucks')
plt.ylim([-.05, 1.05])
# plt.xlim([0,1300])
plt.legend()

fig = plt.figure()
ax = plt.tricontour(data_df['Vehicles'], data_df['Prop Trucks'], data_df['Redux Cruising'], cmap='tab10')
cb = fig.colorbar(ax)
plt.title('Reduction in Cruising (min) between FCFS and MPC')
plt.xlabel('Number of Vehicles')
plt.ylabel('Proportion of Trucks')
plt.ylim([-.05, 1.05])
# plt.xlim([0,1300])
plt.legend()