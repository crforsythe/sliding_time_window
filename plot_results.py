import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as p
import pickle
from collections import OrderedDict
def load_results():
    fileName = 'Results/results.dat'
    with open(fileName, 'rb') as file:
        r = pickle.load(file)
        file.close()

    return r

def get_clean_results(results_dict):
    rows = []
    for num_trips, importance_dict in results_dict.items():
        for importance_level, value_dict in importance_dict.items():
            output_df = value_dict['output_df']
            objective_dict = value_dict['objective_dict']
            no_park_entities = output_df.loc[:, 'Assigned']!='Yes'
            park_entities = output_df.loc[:, 'Assigned'] == 'Yes'
            num_assigned = len(output_df.loc[park_entities, :])
            num_unassigned = len(output_df.loc[no_park_entities, :])
            double_park_minutes = np.sum(output_df.loc[no_park_entities, 'Double Park']*output_df.loc[no_park_entities, 's_i'])
            cruising_minutes = np.sum(
                output_df.loc[no_park_entities, 'Cruising'] * output_df.loc[no_park_entities, 'Potential Cruising Time'])
            double_park_vehicles = np.sum(
                output_df.loc[no_park_entities, 'Double Park'])
            cruising_vehicles = np.sum(
                output_df.loc[no_park_entities, 'Cruising'])
            row = [num_trips, objective_dict['weights'][0], objective_dict['weights'][1], num_assigned, num_unassigned, double_park_minutes, cruising_minutes, double_park_vehicles, cruising_vehicles]
            rows.append(row)

    data = np.array(rows)

    df = p.DataFrame(data, columns=['Total # of Requests', 'Double-Park Vehicle-Minutes Weight', 'Cruising Vehicle-Minutes Weight', 'Vehicles Assigned', 'Vehicle Unassigned', 'Double Park Vehicle-Minutes', 'Cruising Vehicle-Minutes', 'Double Park Vehicles', 'Cruising Vehicles'])

    return df

t = load_results()
tClean = get_clean_results(t)

x = 5
gr = (1+np.sqrt(5))/2
y = x/gr

fig, ax = plt.subplots(1,1)
ax = sns.scatterplot(x='Double Park Vehicle-Minutes', y='Cruising Vehicle-Minutes', data=tClean, hue='Total # of Requests', palette=sns.color_palette(n_colors=len(set(tClean.loc[:, 'Total # of Requests']))))
sns.move_legend(ax, bbox_to_anchor=(1,1), loc='upper left')
plt.savefig('Plots/MultiObjectiveVehicleMinutes.png', bbox_inches='tight')
plt.show()

fig, ax = plt.subplots(1,1)
ax = sns.scatterplot(x='Double Park Vehicles', y='Cruising Vehicles', data=tClean, hue='Total # of Requests', palette=sns.color_palette(n_colors=len(set(tClean.loc[:, 'Total # of Requests']))))
sns.move_legend(ax, bbox_to_anchor=(1,1), loc='upper left')
plt.savefig('Plots/MultiObjectiveVehicles.png', bbox_inches='tight')
plt.show()