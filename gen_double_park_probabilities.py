import pandas as p
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
def loadData(large=False):

    if(large):
        file = '/Users/connorforsythe/Dropbox/CMU/SmartCurbs/Street Sense Data/All parking data delivery 9 Apr 2022/all_parking_context_data_9_April_2022.csv'
        file = 'C:/Users/Aaron/Documents/GitHub/sliding_time_window_data/Data/Street Sense Data/All parking data delivery 9 Apr 2022/all_parking_context_data_9_April_2022.csv'
    else:
        file = '/Users/connorforsythe/Dropbox/CMU/SmartCurbs/Street Sense Data/All parking data delivery 9 Apr 2022/all_parking_data_9_April_2022.csv'
        file = 'C:/Users/Aaron/Documents/GitHub/sliding_time_window_data/Data/Street Sense Data/All parking data delivery 9 Apr 2022/all_parking_data_9_April_2022.csv'
    data = p.read_csv(file)
    return data

def estimateDoubleParkProbability(length, data):
    subData = data.loc[data.loc[:, 'parking_status']=='double_parked']
    totalN = len(subData)
    numGreater = len(subData.loc[subData.loc[:, 'duration']>=length, :])

    return numGreater/totalN

def get_double_park_probability(length):
    data = loadData()
    return estimateDoubleParkProbability(length, data)

def getDoubleParkProbabilities():
    t = loadData()
    lengths = np.arange(0, 6001, 1)
    rows = []
    for length in tqdm(lengths):
        row = [length, estimateDoubleParkProbability(length, t)]
        rows.append(row)

    df = p.DataFrame(np.array(rows), columns=['Parking Length (seconds)', 'Probability of Double Parking'])
    return df
if __name__=='__main__':
    t = loadData()



    x = 5
    gr = (1+np.sqrt(5))/2
    y = x/gr

    fig, ax = plt.subplots(1,1,figsize=(x,y))
    ax = sns.histplot(x='duration', hue='parking_status', data=t)
    plt.xlim([0, 6000])
    plt.xlabel('Parking Length (seconds)')
    plt.savefig('Plots/DoubleParkingHist.png', bbox_inches='tight')


    dp = getDoubleParkProbabilities()

    fig, ax = plt.subplots(1,1,figsize=(x,y))
    ax = sns.lineplot(x='Parking Length (seconds)', y='Probability of Double Parking', data=dp)
    plt.savefig('Plots/DoubleParkingProb.png', bbox_inches='tight')
    plt.show()