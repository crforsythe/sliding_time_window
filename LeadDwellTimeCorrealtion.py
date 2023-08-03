import pandas as p
from cfImplementation import load_nhts_data
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

a = load_nhts_data()

x = 6
gr = (1+np.sqrt(5))/2
y = x/gr

fig, ax = plt.subplots(1,1,figsize=(x,y))
ax = sns.scatterplot(x='Travel Time', y='Dwell Time', data=a)
plt.show()