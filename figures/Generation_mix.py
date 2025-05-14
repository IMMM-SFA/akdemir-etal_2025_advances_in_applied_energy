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
gen_types = ['Coal', 'Biomass', 'Geothermal', 'Oil', 'Nuclear', 'Gas', 'Hydro', 'Solar', 'Wind', 'OffshoreWind', 'Storage']
gen_types_mwh = ['Coal', 'Biomass', 'Geothermal', 'Oil', 'Gas', 'Hydro', 'Solar', 'Wind', 'OffshoreWind']
##########################################

#Reading nodal topology
all_nodes = pd.read_csv('../../Supplementary_Data/BA_Topology_Files/selected_nodes_125.csv', header=0)
all_node_numbers = [*all_nodes['SelectedNodes']]
all_node_strings = [f'bus_{i}' for i in all_node_numbers]
topology_10k_nodes = pd.read_csv('../../Supplementary_Data/BA_Topology_Files/10k_Load.csv', header=0)
nodal_info = pd.read_csv('../../Supplementary_Data/BA_Topology_Files/Nodal_information.csv', header=0)

#Reading generation and outage data and calculating yearly generation by type in percentages
for sc in scenarios:

    #Creating dictionaries to store yearly percentages
    globals()[f'{sc}_generation'] = {}

    for type in gen_types:
        globals()[f'{sc}_generation'][type] = []

    #Reading necessary input and outputs
    gen_outage_inputs = pd.read_csv(f'Z:/im3/exp_b/exp_b_multi_model_coupling_west/experiment_runs/{run_name}/{sc}/go/input/lost_capacity_csv/lost_capacity_2020.csv', header=0)

    for year in years:

        for r in glob.glob(f'Z:/im3/exp_b/exp_b_multi_model_coupling_west/experiment_runs/{run_name}/{sc}/go/output/native_output/{year}/mwh/mwh_{year}PI*.parquet'):
            gen_result = pd.read_parquet(r)

        for t in glob.glob(f'Z:/im3/exp_b/exp_b_multi_model_coupling_west/experiment_runs/{run_name}/{sc}/go/output/native_output/{year}/storage_discharge/storage_discharge_{year}PI*.parquet'):
            storage_result = pd.read_parquet(t)

        nuclear_input = pd.read_csv(f'Z:/im3/exp_b/exp_b_multi_model_coupling_west/experiment_runs/{run_name}/{sc}/go/input/must_run_csv/{year}/must_run_{year}PI.csv', header=0)
        thermal_input = pd.read_csv(f'Z:/im3/exp_b/exp_b_multi_model_coupling_west/experiment_runs/{run_name}/{sc}/go/input/thermal_generators_csv/{year}/thermal_gens_{year}PI.csv', header=0)

        #Calculating generation mix and saving it into related lists

        Total_Storage = storage_result['Value'].sum()
        if Total_Storage < 0:
            Total_Storage = 0

        Total_Nuclear = nuclear_input.sum().sum()*8760
        if Total_Nuclear == 0:
            pass
        
        else:
            Total_Nuclear = Total_Nuclear - (gen_outage_inputs['Nuclear_ovr_1000'].sum()/len(thermal_input.loc[thermal_input['Fuel']=='NUC (Nuclear)']))
            if Total_Nuclear < 0:
                Total_Nuclear = 0

        for s_type in gen_types_mwh:
            globals()[f'Total_{s_type}'] = gen_result.loc[gen_result['Type']==s_type]['Value'].sum()

        Total_Generation = Total_Coal + Total_Biomass + Total_Geothermal + Total_Oil + Total_Nuclear + Total_Gas + Total_Hydro + Total_Solar + Total_Wind + Total_OffshoreWind + Total_Storage

        for s_type in gen_types:
            globals()[f'{sc}_generation'][s_type].append(round(globals()[f'Total_{s_type}']/Total_Generation*100, 2))

            

############## Initializing the figure ##############

# Creating the outer 4 panel layout of the figure
plt.rcParams['font.sans-serif'] = "Arial"
plt.rcParams.update({'font.size': 12.5})

fig = plt.figure(figsize=(20, 11.4))
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
inner_axis_width_1 = "80%"
inner_axis_height_1 = "70%"

ax1_in1 = inset_axes(ax1, width=inner_axis_width_1, height=inner_axis_height_1, loc='upper center', borderpad=0.8)
ax2_in1 = inset_axes(ax2, width=inner_axis_width_1, height=inner_axis_height_1, loc='upper center', borderpad=0.8)
ax3_in1 = inset_axes(ax3, width=inner_axis_width_1, height=inner_axis_height_1, loc='upper center', borderpad=0.8)
ax4_in1 = inset_axes(ax4, width=inner_axis_width_1, height=inner_axis_height_1, loc='upper center', borderpad=0.8)

############## Plotting generation mixes ##############

handles = []
patch1 = Patch(facecolor='#949494', edgecolor='black',label='Coal',linewidth=0.5)
patch2 = Patch(facecolor='#FBAFE4', edgecolor='black',label='Biomass',linewidth=0.5)
patch3 = Patch(facecolor='#808000', edgecolor='black',label='Geothermal',linewidth=0.5)
patch4 = Patch(facecolor='#000000', edgecolor='black',label='Oil',linewidth=0.5)
patch5 = Patch(facecolor='#D55E00', edgecolor='black',label='Nuclear',linewidth=0.5)
patch6 = Patch(facecolor='#CA9161', edgecolor='black',label='Natural Gas',linewidth=0.5)
patch7 = Patch(facecolor='#0173B2', edgecolor='black',label='Hydro',linewidth=0.5)
patch8 = Patch(facecolor='#ECE133', edgecolor='black',label='Solar',linewidth=0.5)
patch9 = Patch(facecolor='#029E73', edgecolor='black',label='Onshore Wind',linewidth=0.5)
patch10 = Patch(facecolor='#56B4E9', edgecolor='black',label='Offshore Wind',linewidth=0.5)
patch11 = Patch(facecolor='#CC78BC', edgecolor='black',label='Storage',linewidth=0.5)
handles.extend([patch1,patch2,patch3,patch4,patch5,patch6,patch7,patch8,patch9,patch10,patch11])

for idx, sc in enumerate(scenarios):

    globals()[f'ax{idx+1}_in1'].stackplot(years, globals()[f'{sc}_generation']['Coal'], globals()[f'{sc}_generation']['Biomass'], globals()[f'{sc}_generation']['Geothermal'], globals()[f'{sc}_generation']['Oil'], globals()[f'{sc}_generation']['Nuclear'], 
                                          globals()[f'{sc}_generation']['Gas'], globals()[f'{sc}_generation']['Hydro'], globals()[f'{sc}_generation']['Solar'], globals()[f'{sc}_generation']['Wind'], globals()[f'{sc}_generation']['OffshoreWind'], 
                                          globals()[f'{sc}_generation']['Storage'], colors=['#949494','#FBAFE4','#808000','#000000','#D55E00','#CA9161','#0173B2','#ECE133','#029E73','#56B4E9','#CC78BC'])
    globals()[f'ax{idx+1}_in1'].set_ylabel('Normalized Annual Generation (%)', weight = 'bold', fontsize=13.5)
    globals()[f'ax{idx+1}_in1'].set_xlim([years[0],years[-1]])
    globals()[f'ax{idx+1}_in1'].set_ylim([0,100])
    globals()[f'ax{idx+1}_in1'].set_yticks([0,10,20,30,40,50,60,70,80,90,100])
    globals()[f'ax{idx+1}_in1'].set_xticks(years)

    globals()[f'ax{idx+1}_in1'].axvline(x=years[1], color='black',linewidth=1)
    globals()[f'ax{idx+1}_in1'].axvline(x=years[2], color='black',linewidth=1)
    globals()[f'ax{idx+1}_in1'].axvline(x=years[3], color='black',linewidth=1)
    globals()[f'ax{idx+1}_in1'].axvline(x=years[4], color='black',linewidth=1)
    globals()[f'ax{idx+1}_in1'].axvline(x=years[5], color='black',linewidth=1)
    globals()[f'ax{idx+1}_in1'].axvline(x=years[6], color='black',linewidth=1)

    secay = globals()[f'ax{idx+1}_in1'].secondary_yaxis('right')
    globals()[f'ax{idx+1}_in1'].tick_params(axis='x', which='both', length=8)
    globals()[f'ax{idx+1}_in1'].tick_params(axis='y', which='both', length=8)
    secay.tick_params(axis='y', which='both', length=8)
    secay.set_yticks([0,10,20,30,40,50,60,70,80,90,100])

    globals()[f'ax{idx+1}_in1'].legend(handles=handles,loc='center', bbox_to_anchor=(0.5, -0.25), ncol=6, fontsize=9.5)

plt.savefig(f'Generation_mix_{t_scenario}_scenarios.png', dpi=400, bbox_inches='tight')
plt.show()
plt.clf()
