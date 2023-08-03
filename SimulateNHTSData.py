import pandas as p
import numpy as np
from gen_double_park_classification import apply_double_park_classification, apply_potential_cruising_time
from execute_v2_cf import run_optimization, deepcopy
from collections import OrderedDict
import pickle
from gen_veh_arrivals import gen_veh_arrivals as gen_truck_arrivals
def load_raw_nhts_data(useCols=None):
    file = 'Data/NHTS/csv/trippub.csv'
    if(useCols==None):
        data = p.read_csv(file)
    else:
        data = p.read_csv(file, usecols=useCols)
    return data

def load_nhts_data():

    tripDistCol = 'VMT_MILE'
    tripReasonCol = 'WHYTO'
    dwellTimeCol = 'DWELTIME'
    tripTimeCol = 'TRVLCMIN'
    tripModeCol = 'TRPTRANS'
    tripWeightCol = 'WTTRDFIN'
    urbanSizeCol = 'URBANSIZE'
    startTimeCol = 'STRTTIME'
    endTimeCol = 'ENDTIME'
    personalModes = [3,4,5,6]
    taxiModes = [18]

    useCols = [tripDistCol, tripReasonCol, dwellTimeCol, tripTimeCol, tripModeCol, tripWeightCol, urbanSizeCol, startTimeCol, endTimeCol]
    r = load_raw_nhts_data(useCols)

    r = r.loc[(r.loc[:, tripModeCol].isin(personalModes))|(r.loc[:, tripModeCol].isin(taxiModes)), :]

    for col in r.columns:
        r = r.loc[r.loc[:, col]>=0, :]

    r = r.loc[r.loc[:, tripReasonCol]>3, :]

    reasonReplaceDict = dict(zip([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 97],
                                 ['Regular home activities', 'Work from home', 'Work', 'Work-related trip', 'Volunteer activities', 'Drop off /pick up someone', 'Change type of transportation', 'Attend school as a student', 'Attend child care', 'Attend adult care', 'Buy goods', 'Buy services', 'Buy meals', 'Other general errands', 'Recreational activities', 'Exercise', 'Visit friends or relatives', 'Health care visit', 'Religious or other community activities', 'Something else']))

    r.loc[:, tripReasonCol] = r.loc[:, tripReasonCol].replace(reasonReplaceDict)

    cleanNames = ['Trip Distance', 'Destination Reason', 'Dwell Time', 'Travel Time', 'Mode', 'Weight', 'Urban Size', 'Start Time', 'End Time']
    r = r.rename(dict(zip(useCols, cleanNames)), axis=1)

    r = r.loc[r.loc[:, 'Urban Size']<6, :]
    # r = convert_to_day_minutes(r)
    r = r.loc[r.loc[:, 'Start Time']>=(7*60), :]
    r = r.loc[r.loc[:, 'Start Time'] <= (18 * 60), :]
    r = r.loc[r.loc[:, 'End Time'] <= (18 * 60), :]
    r = apply_probabilities(r)
    return r


def select_n_trips(data, num_sample=50, replace=True):
    num_rows = len(data)

    if(num_rows<num_sample):
        num_sample = num_rows
        return data
    else:
        indicies = np.random.choice(num_rows, num_sample, p=data.loc[:, 'p'], replace=replace)
        sub_data = data.iloc[indicies, :]
        return sub_data

def apply_probabilities(data):
    weight_col = 'Weight'
    data.loc[:, 'p'] = data.loc[:, weight_col] / np.sum(data.loc[:, weight_col])
    return data

def construct_truth_dataframe(data, phi=10, b_shift=10, receivedDelta=30):
    data.loc[:, 'Received'] = data.loc[:, 'Start Time']
    data.loc[:, 'Received_OG'] = data.loc[:, 'Start Time']
    data.loc[:, 'Received-Diff'] = data.loc[:, 'End Time']-data.loc[:, 'Start Time']
    data.loc[data.loc[:, 'Received-Diff']>receivedDelta, 'Received'] = data.loc[data.loc[:, 'Received-Diff']>receivedDelta, 'End Time']-receivedDelta
    data.loc[:, 'a_i_OG'] = data.loc[:, 'End Time']
    data.loc[:, 'b_i_OG'] = data.loc[:, 'a_i_OG']+b_shift
    data.loc[:, 's_i'] = data.loc[:, 'Dwell Time']
    data.loc[:, 'd_i_OG'] = data.loc[:, 'End Time']+data.loc[:, 'Dwell Time']
    data.loc[:, 'phi'] = phi

    data = apply_double_park_classification(data)
    data = apply_potential_cruising_time(data)
    data = data.sort_values('a_i_OG')
    data.loc[:, 'Vehicle'] = range(len(data))
    data.index = range(len(data))
    data.loc[:, 'Type'] = 'Passenger'
    return data

def convert_to_day_minutes(data):
    start_col = 'Start Time'
    end_col = 'End Time'

    start_mod = np.mod(data.loc[:, start_col], 100)
    end_mod = np.mod(data.loc[:, end_col], 100)

    data.loc[:, start_col] = data.loc[:, start_col] - start_mod
    data.loc[:, end_col] = data.loc[:, end_col] - end_mod

    data.loc[:, start_col] = data.loc[:, start_col]/100*60
    data.loc[:, end_col] = data.loc[:, end_col]/100*60

    data.loc[:, start_col] = data.loc[:, start_col] + start_mod
    data.loc[:, end_col] = data.loc[:, end_col] + end_mod

    data.loc[:, start_col] = data.loc[:, start_col].astype(int)
    data.loc[:, end_col] = data.loc[:, end_col].astype(int)

    return data


def join_requests(passenger, truck, phi=5, receivedDelta=30):
    truck.loc[:, 'No-Park Outcome'] = 'Double-Park'
    truck.loc[:, 'Expected Double Park'] = 1
    truck.loc[:, 'Actual Double Park'] = 1

    truck.loc[:, 'Expected Cruising'] = 0
    truck.loc[:, 'Actual Cruising'] = 0

    truck.loc[:, 'Expected Cruising Time'] = 0
    truck.loc[:, 'Actual Cruising Time'] = 0

    truck.loc[:, 'Received'] = truck.loc[:, 'a_i_OG']-receivedDelta
    truck.loc[:, 'a_i_OG'] = truck.loc[:, 'a_i_OG']
    truck.loc[:, 'd_i_OG'] = truck.loc[:, 'd_i_OG']
    truck.loc[:, 'phi'] = phi
    r = p.concat([passenger, truck])
    r.loc[:, 'Vehicle'] = range(len(r))
    r.index = range(len(r))
    return r


if __name__=='__main__':
    np.random.seed(8131970)
    data = load_nhts_data()
    t_sample = select_n_trips(data, num_sample=50)
    # t_sample = select_n_trips(data, num_sample=20)
    # t = construct_truth_dataframe(t_sample)
    #
    # truck = gen_truck_arrivals(50)
    #
    # j = join_requests(t, truck)
    # w = 50
    # objective_dict = {'parked': [0, 0], 'weights': [w, 100-w], 'cols': [['Expected Double Park', 's_i'], ['Expected Cruising', 'Expected Cruising Time']]}
    # req_master = run_optimization(deepcopy(t), objective_dict, buffer=0)

    # objective_dict = {'parked': [0, 0], 'weights': [w, 100-w], 'cols': [['Expected Double Park', 's_i'], ['Expected Cruising', 'Expected Cruising Time']]}
    # req_master = run_optimization(deepcopy(jointData), objective_dict)

    # r = OrderedDict()
    # badKeys = []
    # for i in np.arange(50,151, 100):
    #     r[i] = OrderedDict()
    #     t_sample = select_n_trips(data, num_sample=i)
    #     for jointData in np.arange(1, 100, 100):
    #         try:
    #
    #             t = construct_truth_dataframe(t_sample)
    #             objective_dict = {'parked': [0, 0], 'weights': [jointData, 100-jointData],
    #                               'cols': [['Double Park', 's_i'], ['Cruising', 'Potential Cruising Time']]}
    #             req_master = run_optimization(deepcopy(t), objective_dict)
    #             r[i][jointData] = OrderedDict()
    #             r[i][jointData]['input_df'] = t
    #             r[i][jointData]['objective_dict'] = objective_dict
    #             r[i][jointData]['output_df'] = req_master
    #         except:
    #             badKeys.append([i, jointData])
    #
    #
    # # with open('Results/results.dat', 'wb') as file:
    # #     pickle.dump(r, file)
    # #     file.close()
    #
    # #
    # #
    #
    # # objective_dict = {'parked':[0,0], 'weights':[1, 100], 'cols':[['Double Park', 's_i'], ['Cruising', 'Potential Cruising Time']]}
    # # req_master_2 = run_optimization(deepcopy(t), objective_dict)
