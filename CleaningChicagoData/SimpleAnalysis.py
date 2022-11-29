import pandas as p
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def loadData():
    file = 'Data/ChicagoTripTimeData.csv'
    data = p.read_csv(file, usecols=['Trip Seconds'])
    return data

data = loadData()
x = 5
gr = (1+np.sqrt(5))/2
y = x/gr
fig, ax = plt.subplots(1,1,figsize=(x,y))

ax = sns.histplot(data=data, x='Trip Seconds')
plt.xlim([0, 6000])
plt.savefig('Plots/TripTimeHistogram.png', bbox_inches='tight', dpi=200)
plt.show()