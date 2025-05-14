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

##########################################
years = [2020, 2025, 2030, 2035, 2040, 2045, 2050, 2055]
t_scenario = 'cooler'
run_name = 'run_110824'
scenarios = [f'rcp45{t_scenario}_ssp3', f'rcp85{t_scenario}_ssp3', f'rcp45{t_scenario}_ssp5', f'rcp85{t_scenario}_ssp5']
##########################################

#Reading nodal topology
all_nodes = pd.read_csv('../../Supplementary_Data/BA_Topology_Files/selected_nodes_125.csv', header=0)
all_node_numbers = [*all_nodes['SelectedNodes']]
all_node_strings = [f'bus_{i}' for i in all_node_numbers]

hourly_timestamp = pd.date_range(start='2015-01-01 00:00:00', end='2015-12-31 23:00:00', freq='h')

for sc in scenarios:

    #Creating dataframes to store data side by side
    globals()[f'{sc}_storage_cycle'] = pd.DataFrame(np.zeros((24,len(years))), columns=years)
    globals()[f'{sc}_net_demand'] = pd.DataFrame(np.zeros((24,len(years))), columns=years)

    for year in years:

        for r in glob.glob(f'Z:/im3/exp_b/exp_b_multi_model_coupling_west/experiment_runs/{run_name}/{sc}/go/output/native_output/{year}/mwh/mwh_{year}PI*.parquet'):
            gen_result = pd.read_parquet(r)

        for d in glob.glob(f'Z:/im3/exp_b/exp_b_multi_model_coupling_west/experiment_runs/{run_name}/{sc}/go/input/nodal_load_csv/{year}/nodal_load_*.csv'):
            demand = pd.read_csv(d, header=0)

        for s1 in glob.glob(f'Z:/im3/exp_b/exp_b_multi_model_coupling_west/experiment_runs/{run_name}/{sc}/go/output/native_output/{year}/storage_discharge/storage_discharge_{year}PI*.parquet'):
            storage_discharge = pd.read_parquet(s1)

        for s2 in glob.glob(f'Z:/im3/exp_b/exp_b_multi_model_coupling_west/experiment_runs/{run_name}/{sc}/go/output/native_output/{year}/storage_charge/storage_charge_{year}PI*.parquet'):
            storage_charge = pd.read_parquet(s2)
            
        storage_params = pd.read_csv(f'Z:/im3/exp_b/exp_b_multi_model_coupling_west/experiment_runs/{run_name}/{sc}/go/input/storage_parameters_csv/{year}/storage_params_{year}PI.csv')

        #Calculating storage utilization and saving it into relevant dataframe
        total_discharge_capacity = storage_params['discharge_rate'].sum()
        total_charge_capacity = storage_params['charge_rate'].sum()

        all_discharge = storage_discharge.groupby('Time').sum(numeric_only=True)['Value'].values/total_discharge_capacity*100
        all_charge = storage_charge.groupby('Time').sum(numeric_only=True)['Value'].values/total_charge_capacity*100
        storage_trend = all_discharge - all_charge
        storage_trend = pd.DataFrame(storage_trend, index=hourly_timestamp)
        globals()[f'{sc}_storage_cycle'].loc[:,year] = storage_trend.groupby([storage_trend.index.hour]).mean().values.reshape(-1)

        #Calculating net load and saving it into relevant dataframe
        all_renewable_generation = gen_result.loc[gen_result['Type'].isin(['Solar', 'Wind', 'OffshoreWind'])].groupby('Time').sum(numeric_only=True)['Value'].values
        all_demand = demand.sum(axis=1).values
        net_demand = all_demand - all_renewable_generation
        net_demand = pd.DataFrame(net_demand, index=hourly_timestamp)
        net_demand = net_demand.groupby([net_demand.index.hour]).mean().values.reshape(-1)
        net_demand = net_demand/np.max(net_demand)*100
        globals()[f'{sc}_net_demand'].loc[:,year] = net_demand

        

############## Initializing the figure ##############

# Creating the outer 4 panel layout of the figure
plt.rcParams['font.sans-serif'] = "Arial"
plt.style.use('seaborn-v0_8-whitegrid') 
plt.rcParams.update({'font.size': 12})

fig = plt.figure(figsize=(24, 12))
gs = gridspec.GridSpec(2, 2, wspace=0, hspace=0)

ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[1, 0])
ax4 = fig.add_subplot(gs[1, 1])

ax1.set_title("RCP 4.5", weight='bold', fontsize=18)
ax2.set_title("RCP 8.5", weight='bold', fontsize=18)
ax1.set_ylabel("SSP 3", weight='bold', fontsize=18)
ax3.set_ylabel("SSP 5", weight='bold', fontsize=18)

for ax in [ax1, ax2, ax3, ax4]:
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(1.5)

# Creating lower level inner axes as inset axes
inner_axis_width_1 = "43%"
inner_axis_height_1 = "84%"
inner_axis_width_2 = "43%"
inner_axis_height_2 = "84%"

ax1_in1 = inset_axes(ax1, width=inner_axis_width_1, height=inner_axis_height_1, loc='upper left', borderpad=1.5, bbox_to_anchor=(0.07, 0.02, 0.9, 0.98), bbox_transform=ax1.transAxes)
ax1_in2 = inset_axes(ax1, width=inner_axis_width_2, height=inner_axis_height_2, loc='upper left', borderpad=1.5, bbox_to_anchor=(0.57, 0.02, 0.9, 0.98), bbox_transform=ax1.transAxes)

ax2_in1 = inset_axes(ax2, width=inner_axis_width_1, height=inner_axis_height_1, loc='upper left', borderpad=1.5, bbox_to_anchor=(0.07, 0.02, 0.9, 0.98), bbox_transform=ax2.transAxes)
ax2_in2 = inset_axes(ax2, width=inner_axis_width_2, height=inner_axis_height_2, loc='upper left', borderpad=1.5, bbox_to_anchor=(0.57, 0.02, 0.9, 0.98), bbox_transform=ax2.transAxes)

ax3_in1 = inset_axes(ax3, width=inner_axis_width_1, height=inner_axis_height_1, loc='upper left', borderpad=1.5, bbox_to_anchor=(0.07, 0.02, 0.9, 0.98), bbox_transform=ax3.transAxes)
ax3_in2 = inset_axes(ax3, width=inner_axis_width_2, height=inner_axis_height_2, loc='upper left', borderpad=1.5, bbox_to_anchor=(0.57, 0.02, 0.9, 0.98), bbox_transform=ax3.transAxes)

ax4_in1 = inset_axes(ax4, width=inner_axis_width_1, height=inner_axis_height_1, loc='upper left', borderpad=1.5, bbox_to_anchor=(0.07, 0.02, 0.9, 0.98), bbox_transform=ax4.transAxes)
ax4_in2 = inset_axes(ax4, width=inner_axis_width_2, height=inner_axis_height_2, loc='upper left', borderpad=1.5, bbox_to_anchor=(0.57, 0.02, 0.9, 0.98), bbox_transform=ax4.transAxes)

############## Plotting renewable curtailment and storage decision trends ##############

# colors = ['#D0F0C0', '#B2E0A5', '#95D089', '#77C06E', '#59B052', '#3BA037', '#1E901B', '#008000']
colors = ['#E9F6E5', '#C8DEC9', '#A6C6AC', '#85AE90', '#649674', '#437E58', '#21663B', '#004E1F']

for idx, sc in enumerate(scenarios):

    for idx2, yy in enumerate(years):

        globals()[f'ax{idx+1}_in1'].plot(range(0,24), globals()[f'{sc}_storage_cycle'][yy], label=yy, color=colors[idx2])
        globals()[f'ax{idx+1}_in2'].plot(range(0,24), globals()[f'{sc}_net_demand'][yy], label=yy, color=colors[idx2])

    globals()[f'ax{idx+1}_in1'].set_xticks([0,2,4,6,8,10,12,14,16,18,20,22], ['12\nAM','2\nAM','4\nAM','6\nAM','8\nAM','10\nAM','12\nPM','2\nPM','4\nPM','6\nPM','8\nPM','10\nPM'], fontsize=12)
    globals()[f'ax{idx+1}_in2'].set_xticks([0,2,4,6,8,10,12,14,16,18,20,22], ['12\nAM','2\nAM','4\nAM','6\nAM','8\nAM','10\nAM','12\nPM','2\nPM','4\nPM','6\nPM','8\nPM','10\nPM'], fontsize=12)
    globals()[f'ax{idx+1}_in1'].set_xlim([0,23])
    globals()[f'ax{idx+1}_in2'].set_xlim([0,23])

    globals()[f'ax{idx+1}_in1'].set_yticks([-25,-20,-15,-10,-5,0,5,10,15,20,25], [-25,-20,-15,-10,-5,0,5,10,15,20,25], fontsize=12)
    globals()[f'ax{idx+1}_in2'].set_yticks([-20,-10,0,10,20,30,40,50,60,70,80,90,100,110], [-20,-10,0,10,20,30,40,50,60,70,80,90,100,110], fontsize=12)
    
    globals()[f'ax{idx+1}_in1'].set_xlabel('')
    globals()[f'ax{idx+1}_in2'].set_xlabel('')
    globals()[f'ax{idx+1}_in1'].set_ylabel('Percentage of Total Storage Capacity Utilized\n(Negative = Charge, Positive = Discharge)', weight='bold', fontsize=12.5)
    globals()[f'ax{idx+1}_in2'].set_ylabel('Percentage of Net Demand to Maximum Daily\nNet Demand (Negative = Solar + Wind > Demand)', weight='bold', fontsize=12.5)
    
    globals()[f'ax{idx+1}_in1'].grid(axis='both', color='#949494', alpha=0.3)
    globals()[f'ax{idx+1}_in2'].grid(axis='both', color='#949494', alpha=0.3)

    globals()[f'ax{idx+1}_in1'].tick_params(axis='both', which='both', length=10, color='#CCCCCC')
    globals()[f'ax{idx+1}_in2'].tick_params(axis='both', which='both', length=10, color='#CCCCCC')

    globals()[f'ax{idx+1}_in1'].legend(loc='upper left', ncol=2, fontsize=12, frameon=True, framealpha=1)

plt.savefig(f'Storage_netdemand_trends_{t_scenario}_scenarios.png', dpi=300, bbox_inches='tight')
plt.show()
plt.clf()
