import pandas as p
import numpy as np
from scipy.stats import norm
def apply_double_park_classification(df):
    double_park_col = []

    for ind, row in df.iterrows():
        try:
            if(row['Assigned']=='Yes'):
                double_park_col.append('Parked')
            else:
                double_park_col.append(get_output_double_park_outcome(row['Parking Length']))
        except:
            double_park_col.append(get_output_double_park_outcome(row['s_i']))

    df.loc[:, 'No-Park Outcome'] = double_park_col
    df.loc[:, 'Double Park'] = 0
    df.loc[:, 'Cruising'] = 0

    df.loc[df.loc[:, 'No-Park Outcome'] == 'Double Park', 'Double Park'] = 1
    df.loc[df.loc[:, 'No-Park Outcome']=='Cruising', 'Cruising'] = 1

    return df

def apply_potential_cruising_time(df):

    df.loc[:, 'Potential Cruising Time'] = get_cruising_length(df)

    return df

def get_output_double_park_outcome(park_length):
    random_number = simulate_output_double_park_random_number(park_length)
    double_park_probability = get_probability_double_park_outcome(park_length)
    if(random_number<double_park_probability):
        return 'Double Park'
    else:
        return 'Cruising'

def simulate_output_double_park_random_number(park_length):

    return np.random.uniform()

def simulate_output_cruising_random_number(park_length):

    return np.random.uniform()

def get_cruising_length(df):
    return np.random.normal(3, 1, len(df))



def get_probability_double_park_outcome(park_length):
    if(park_length<10):
        return 1
    else:
        return 0