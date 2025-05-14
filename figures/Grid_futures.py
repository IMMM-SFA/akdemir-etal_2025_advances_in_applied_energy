import pandas as pd
import numpy as np
import geopandas as gpd
import yaml
from shapely.geometry import Point
from matplotlib.patches import Patch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import glob
import seaborn as sns

##########################################
years = [2020, 2025, 2030, 2035, 2040, 2045, 2050, 2055]
t_scenario = 'cooler'
run_name = 'run_110824'
scenarios = [f'rcp45{t_scenario}_ssp5', f'rcp85{t_scenario}_ssp5', f'rcp45{t_scenario}_ssp3', f'rcp85{t_scenario}_ssp3']
##########################################

#Reading nodal topology
all_nodes = pd.read_csv('../../Supplementary_Data/BA_Topology_Files/selected_nodes_125.csv', header=0)
all_node_numbers = [*all_nodes['SelectedNodes']]
all_node_strings = [f'bus_{i}' for i in all_node_numbers]
topology_10k_nodes = pd.read_csv('../../Supplementary_Data/BA_Topology_Files/10k_Load.csv', header=0)
nodal_info = pd.read_csv('../../Supplementary_Data/BA_Topology_Files/Nodal_information.csv', header=0)


for sc in scenarios:
    globals()[f'Dispatchable_GenCap_{sc}'] = []
    globals()[f'Renewable_GenCap_{sc}'] = []
    globals()[f'Intraregional_Transmission_Cap_{sc}'] = []
    globals()[f'Interregional_Transmission_Cap_{sc}'] = []
    globals()[f'Storage_Cap_{sc}'] = []
    globals()[f'Average_Demand_{sc}'] = []
    globals()[f'Max_Demand_{sc}'] = []

    for year in years:

        Generators = pd.read_csv(f'Z:/im3/exp_b/exp_b_multi_model_coupling_west/experiment_runs/{run_name}/{sc}/go/input/generator_parameters_csv/{year}/data_genparams_{year}PI.csv', header=0)
        Mustrun = pd.read_csv(f'Z:/im3/exp_b/exp_b_multi_model_coupling_west/experiment_runs/{run_name}/{sc}/go/input/must_run_csv/{year}/must_run_{year}PI.csv', header=0)
        Transmission = pd.read_csv(f'Z:/im3/exp_b/exp_b_multi_model_coupling_west/experiment_runs/{run_name}/{sc}/go/input/line_parameters_csv/{year}/line_param_{year}TC.csv', header=0)
        Transmission_line_regions = pd.read_csv(f'Z:/im3/exp_b/exp_b_multi_model_coupling_west/experiment_runs/{run_name}/{sc}/tep/input/line_parameters_csv/{year}/line_param_{year-5}TC.csv', header=0)
        Storage = pd.read_csv(f'Z:/im3/exp_b/exp_b_multi_model_coupling_west/experiment_runs/{run_name}/{sc}/go/input/storage_parameters_csv/{year}/storage_params_{year}PI.csv', header=0)
        
        for d in glob.glob(f'Z:/im3/exp_b/exp_b_multi_model_coupling_west/experiment_runs/{run_name}/{sc}/go/input/nodal_load_csv/{year}/nodal_load_*.csv'):
            Demand = pd.read_csv(d, header=0)

        Renewable_cap = Generators.loc[Generators['typ'].isin(['solar', 'wind', 'offshorewind'])]['maxcap'].sum()
        globals()[f'Renewable_GenCap_{sc}'].append(Renewable_cap/1000)

        Dispatchable_cap = Generators.loc[Generators['typ'].isin(['biomass', 'coal', 'ngcc', 'geothermal', 'oil', 'hydro'])]['maxcap'].sum()
        Dispatchable_cap = Dispatchable_cap + Mustrun.sum().sum()
        globals()[f'Dispatchable_GenCap_{sc}'].append(Dispatchable_cap/1000)

        Interregional_lines = Transmission_line_regions.loc[Transmission_line_regions['transmission_type']=='interregional']['line'].values
        Interregional_transmission_cap = Transmission.loc[Transmission['line'].isin(Interregional_lines)]['limit'].sum()
        globals()[f'Interregional_Transmission_Cap_{sc}'].append(Interregional_transmission_cap/1000)

        Intraregional_lines = Transmission_line_regions.loc[Transmission_line_regions['transmission_type']=='regional']['line'].values
        Intraregional_transmission_cap = Transmission.loc[Transmission['line'].isin(Intraregional_lines)]['limit'].sum()
        globals()[f'Intraregional_Transmission_Cap_{sc}'].append(Intraregional_transmission_cap/1000)

        Storage_cap = Storage['discharge_rate'].sum()
        globals()[f'Storage_Cap_{sc}'].append(Storage_cap/1000)

        Average_demand = Demand.sum(axis=1).mean()
        globals()[f'Average_Demand_{sc}'].append(Average_demand/1000)

        Max_demand = Demand.sum(axis=1).max()
        globals()[f'Max_Demand_{sc}'].append(Max_demand/1000)



############## Plotting Generation, Transmission, Storage and Demand Changes ##############

plt.rcParams['font.sans-serif'] = "Arial"
plt.rcParams.update({'font.size': 11.5})
plt.style.use('seaborn-v0_8-whitegrid') 
fig, ax = plt.subplots(2,3, figsize=(15,8)) 

scenario_colors = ['#8BA5E9', '#C7D5EE', '#EACDBE', '#DD8E78']

for idx, sc in enumerate(scenarios):

    if (sc == 'rcp45hotter_ssp5') or (sc == 'rcp45cooler_ssp5'):
        sc_leg = 'RCP 4.5/SSP 5'
    elif (sc == 'rcp85hotter_ssp5') or (sc == 'rcp85cooler_ssp5'):
        sc_leg = 'RCP 8.5/SSP 5'
    elif (sc == 'rcp45hotter_ssp3') or (sc == 'rcp45cooler_ssp3'):
        sc_leg = 'RCP 4.5/SSP 3'
    elif (sc == 'rcp85hotter_ssp3') or (sc == 'rcp85cooler_ssp3'):
        sc_leg = 'RCP 8.5/SSP 3'

    ax[0,0].plot(years, globals()[f'Dispatchable_GenCap_{sc}'], label=sc_leg, color=scenario_colors[idx])
    ax[0,1].plot(years, globals()[f'Renewable_GenCap_{sc}'], label=sc_leg, color=scenario_colors[idx])
    ax[0,2].plot(years, globals()[f'Storage_Cap_{sc}'], label=sc_leg, color=scenario_colors[idx])
    ax[1,0].plot(years, globals()[f'Intraregional_Transmission_Cap_{sc}'], label=sc_leg, color=scenario_colors[idx])
    ax[1,1].plot(years, globals()[f'Interregional_Transmission_Cap_{sc}'], label=sc_leg, color=scenario_colors[idx])
    ax[1,2].plot(years, globals()[f'Average_Demand_{sc}'], label=sc_leg, color=scenario_colors[idx])

ax[0,0].set_ylabel('Dispatchable Generation Capacity (GW)', weight='bold', fontsize=12.5)
ax[0,1].set_ylabel('Renewable Generation Capacity (GW)', weight='bold', fontsize=12.5)
ax[0,2].set_ylabel('Storage Discharge Capacity (GW)', weight='bold', fontsize=12.5)
ax[1,0].set_ylabel('Intraregional Transmission Capacity (GW)', weight='bold', fontsize=12.5)
ax[1,1].set_ylabel('Interregional Transmission Capacity (GW)', weight='bold', fontsize=12.5)
ax[1,2].set_ylabel('Average Hourly Demand (GWh)', weight='bold', fontsize=12.5)

ax[0,0].set_xlim(years[0], years[-1])
ax[0,1].set_xlim(years[0], years[-1])
ax[0,2].set_xlim(years[0], years[-1])
ax[1,0].set_xlim(years[0], years[-1])
ax[1,1].set_xlim(years[0], years[-1])
ax[1,2].set_xlim(years[0], years[-1])

ax[0,0].set_yticks([100,120,140,160,180,200,220,240,260])
ax[0,1].set_yticks([0,50,100,150,200,250,300,350,400,450,500,550])
ax[0,2].set_yticks([0,50,100,150,200,250,300])
ax[1,0].set_yticks([800,825,850,875,900,925,950,975,1000,1025])
ax[1,1].set_yticks([70,75,80,85,90,95,100,105,110,115,120,125])
ax[1,2].set_yticks([80,90,100,110,120,130,140,150,160,170,180])

ax[0,0].tick_params(axis='both', which='both', length=10, color='#CCCCCC')
ax[0,1].tick_params(axis='both', which='both', length=10, color='#CCCCCC')
ax[0,2].tick_params(axis='both', which='both', length=10, color='#CCCCCC')
ax[1,0].tick_params(axis='both', which='both', length=10, color='#CCCCCC')
ax[1,1].tick_params(axis='both', which='both', length=10, color='#CCCCCC')
ax[1,2].tick_params(axis='both', which='both', length=10, color='#CCCCCC')

ax[0,0].legend(loc='upper left', frameon=True, framealpha=1)
ax[0,1].legend(loc='upper left', frameon=True, framealpha=1)
ax[0,2].legend(loc='upper left', frameon=True, framealpha=1)
ax[1,0].legend(loc='upper left', frameon=True, framealpha=1)
ax[1,1].legend(loc='upper left', frameon=True, framealpha=1)
ax[1,2].legend(loc='upper left', frameon=True, framealpha=1)

plt.tight_layout()
plt.savefig(f'Generation_transmission_demand_storage_change_{t_scenario}_scenarios.png', dpi=300, bbox_inches='tight')
plt.show()
plt.clf()
