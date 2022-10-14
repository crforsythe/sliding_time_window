# -*- coding: utf-8 -*-
"""
Created on Tue Oct 11 08:42:58 2022

@author: Aaron
"""



import numpy as np
import pandas as pd
import pickle
import time
import matplotlib.pyplot as plt
import seaborn as sns


tic = time.time()



#optimal w/ buffer, randomly shifted arrivals, reassassed parking schedule
with open('sliding_window_5.pkl', 'rb') as file:
        redux_FCFS_Opt, \
        redux_FCFS_Opt_df, \
        redux_FCFS_sliding, \
        redux_FCFS_sliding_df, \
        redux_Opt_sliding, \
        redux_Opt_sliding_df, \
        arrival_dfs_df, \
        sum_service_df, \
        runtime, \
        buffer, \
        end, \
        i, \
        c, \
        phi \
            = pickle.load(file)
            
            
plt.figure()
x = [0, 1, 2, 3, 4]
plt.scatter(x, redux_FCFS_Opt, label = 'FCFS to Opt')
plt.plot(x, redux_FCFS_Opt, label = 'FCFS to Opt')
plt.scatter(x, redux_FCFS_sliding, label = 'FCFS to Sliding')
plt.plot(x, redux_FCFS_sliding, label = 'FCFS to Sliding')
plt.scatter(x, redux_Opt_sliding, label = 'Opt to Sliding')
plt.plot(x, redux_Opt_sliding, label = 'Opt to Sliding')
plt.legend()
plt.title('Reduction in Double Parking between different methods \n 30 min sliding time window')
plt.xlabel('Iteration')
plt.ylabel('Reduction in minutes of double parking')