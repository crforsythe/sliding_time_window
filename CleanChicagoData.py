import pandas as p
import numpy as np
from tqdm import tqdm
file = r"G:\Other computers\Aaron's Laptop\Documents\GitHub\sliding_time_window\ChicagoTripTimeData.csv"
usecols = ['Trip Start Timestamp', 'Trip End Timestamp', 'Trip Seconds']
data = p.read_csv(file, usecols=usecols, chunksize=1000000)

r = []
for chunk in tqdm(data):
    r.append(chunk)

r = p.concat(r)
r.to_csv('Data/ChicagoTripTimeData.csv', index=False)