import pandas as p
import numpy as np
from scipy.stats import norm
from gen_double_park_probabilities import get_double_park_probability
def apply_double_park_classification(df):
    double_park_col = []
    double_park_prob_col = []

    for ind, row in df.iterrows():
        try:
            if(row['Assigned']=='Yes'):
                double_park_col.append('Parked')
                double_park_prob_col.append(0)
            else:
                tempOut = get_output_double_park_outcome(row['Parking Length'])

        except:
            tempOut = get_output_double_park_outcome(row['s_i'])

        double_park_prob_col.append(tempOut[0])
        double_park_col.append(tempOut[1])

    df.loc[:, 'No-Park Outcome'] = double_park_col
    df.loc[:, 'Expected Double Park'] = double_park_prob_col
    df.loc[:, 'Expected Cruising'] = 1-df.loc[:, 'Expected Double Park']
    df.loc[:, 'Actual Double Park'] = 0
    df.loc[:, 'Actual Cruising'] = 0

    df.loc[df.loc[:, 'No-Park Outcome'] == 'Double Park', 'Actual Double Park'] = 1
    df.loc[df.loc[:, 'No-Park Outcome']=='Cruising', 'Actual Cruising'] = 1

    return df

def apply_potential_cruising_time(df):

    df.loc[:, 'Expected Cruising Time'] = get_cruising_length(df, return_mean=True)
    df.loc[:, 'Actual Cruising Time'] = get_cruising_length(df)

    return df

def get_output_double_park_outcome(park_length):
    random_number = simulate_output_double_park_random_number(park_length)
    double_park_probability = get_probability_double_park_outcome(park_length)
    r = [double_park_probability]
    if(random_number<double_park_probability):
        r.append('Double Park')
    else:
        r.append('Cruising')

    return r

def simulate_output_double_park_random_number(park_length):

    return np.random.uniform()

def simulate_output_cruising_random_number(park_length):

    return np.random.uniform()

def get_cruising_length(df, mean=3, sd=1, return_mean=False):
    if(return_mean):
        return mean
    else:
        return np.random.normal(mean, sd, len(df))



def get_probability_double_park_outcome(park_length):
    return get_double_park_probability(park_length)