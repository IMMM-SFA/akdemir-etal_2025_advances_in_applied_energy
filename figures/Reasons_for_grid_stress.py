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
from matplotlib.ticker import MultipleLocator  

##########################################
years = [2020, 2025, 2030, 2035, 2040, 2045, 2050, 2055]
t_scenario = 'hotter'
run_name = 'run_110824'
scenarios = [f'rcp45{t_scenario}_ssp3', f'rcp85{t_scenario}_ssp3', f'rcp45{t_scenario}_ssp5', f'rcp85{t_scenario}_ssp5']
quantile = 0.99
##########################################

#Reading nodal topology
all_nodes = pd.read_csv('../../Supplementary_Data/BA_Topology_Files/selected_nodes_125.csv', header=0)
all_node_numbers = [*all_nodes['SelectedNodes']]
all_node_strings = [f'bus_{i}' for i in all_node_numbers]

hourly_timestamp = pd.date_range(start='2015-01-01 00:00:00', end='2015-12-31 23:00:00', freq='h')

for sc in scenarios:
    for year in years:

        #Reading model outputs
        for l1 in glob.glob(f'Z:/im3/exp_b/exp_b_multi_model_coupling_west/experiment_runs/{run_name}/{sc}/go/output/native_output/{year}/duals/duals_{year}PI*.parquet'):
            raw_LMP = pd.read_parquet(l1)

        for l2 in glob.glob(f'Z:/im3/exp_b/exp_b_multi_model_coupling_west/experiment_runs/{run_name}/{sc}/go/output/native_output/{year}/flow/flow_{year}PI*.parquet'):
            raw_flow = pd.read_parquet(l2)

        for l3 in glob.glob(f'Z:/im3/exp_b/exp_b_multi_model_coupling_west/experiment_runs/{run_name}/{sc}/go/output/native_output/{year}/slack/slack_{year}PI*.parquet'):
            raw_LOL = pd.read_parquet(l3)

        for l4 in glob.glob(f'Z:/im3/exp_b/exp_b_multi_model_coupling_west/experiment_runs/{run_name}/{sc}/go/output/native_output/{year}/storage_soc/storage_soc_{year}PI*.parquet'):
            raw_SoC = pd.read_parquet(l4)

        #Reading model inputs
        line_params = pd.read_csv(f'Z:/im3/exp_b/exp_b_multi_model_coupling_west/experiment_runs/{run_name}/{sc}/go/input/line_parameters_csv/{year}/line_param_{year}TC.csv', header=0)
        line_names = line_params['line'].values
        storage_params = pd.read_csv(f'Z:/im3/exp_b/exp_b_multi_model_coupling_west/experiment_runs/{run_name}/{sc}/go/input/storage_parameters_csv/{year}/storage_params_{year}PI.csv', header=0)
        gen_params = pd.read_csv(f'Z:/im3/exp_b/exp_b_multi_model_coupling_west/experiment_runs/{run_name}/{sc}/go/input/generator_parameters_csv/{year}/data_genparams_{year}PI.csv', header=0)
        Transmission_line_regions = pd.read_csv(f'Z:/im3/exp_b/exp_b_multi_model_coupling_west/experiment_runs/{run_name}/{sc}/tep/input/line_parameters_csv/{year}/line_param_{year-5}TC.csv', header=0)
        Interregional_lines = Transmission_line_regions.loc[Transmission_line_regions['transmission_type']=='interregional']['line'].values
        Intraregional_lines = Transmission_line_regions.loc[Transmission_line_regions['transmission_type']=='regional']['line'].values

        for d1 in glob.glob(f'Z:/im3/exp_b/exp_b_multi_model_coupling_west/experiment_runs/{run_name}/{sc}/go/input/nodal_load_csv/{year}/nodal_load_*.csv'):
            demand = pd.read_csv(d1, header=0)

        for d2 in glob.glob(f'Z:/im3/exp_b/exp_b_multi_model_coupling_west/experiment_runs/{run_name}/{sc}/go/input/nodal_offshore_wind_csv/{year}/nodal_offshore_wind_{year}PI_*.csv'):
            offshore_wind_av = pd.read_csv(d2, header=0)

        for d3 in glob.glob(f'Z:/im3/exp_b/exp_b_multi_model_coupling_west/experiment_runs/{run_name}/{sc}/go/input/nodal_wind_csv/{year}/nodal_wind_{year}PI_*.csv'):
            onshore_wind_av = pd.read_csv(d3, header=0)

        for d4 in glob.glob(f'Z:/im3/exp_b/exp_b_multi_model_coupling_west/experiment_runs/{run_name}/{sc}/go/input/nodal_solar_csv/{year}/nodal_solar_{year}PI_*.csv'):
            solar_av = pd.read_csv(d4, header=0)

        #Calculating demand weighted hourly LMP
        nodal_demand_weights = demand.sum()/demand.sum().sum()

        WECC_hourly_average_LMP = np.zeros(len(hourly_timestamp))
        for bus in all_node_strings:
            WECC_hourly_average_LMP = WECC_hourly_average_LMP + (raw_LMP.loc[raw_LMP['Bus']==bus]['Value'].values*nodal_demand_weights[bus])

        #Calculating hourly total unserved energy 
        WECC_hourly_total_LOL = raw_LOL.groupby('Time').sum(numeric_only=True)['Value'].values

        #Calculating hourly available energy storage SoC
        WECC_hourly_total_SoC = raw_SoC.groupby('Time').sum(numeric_only=True)['Value'].values
        WECC_hourly_SoC_availability = (WECC_hourly_total_SoC/storage_params['max_SoC'].sum())*100
        #Since SoC is not allowed to be <20%, adjusting SoC percents accordingly between 0% and 100%
        WECC_hourly_SoC_availability = (WECC_hourly_SoC_availability-20)/(100-20)*100

        #Calculating hourly total demand
        WECC_hourly_total_demand = demand.sum(axis=1).values

        #Calculating hourly total renewables capacity factor
        Total_renewable_capacity = gen_params.loc[gen_params['typ'].isin(['solar','wind','offshorewind'])]['maxcap'].sum()
        WECC_hourly_total_renewables = offshore_wind_av.sum(axis=1).values + onshore_wind_av.sum(axis=1).values + solar_av.sum(axis=1).values
        WECC_hourly_total_renewables = WECC_hourly_total_renewables/Total_renewable_capacity*100

        #Calculating hourly congestion percentages (assuming that lines connecting any two neighoring nodes that are connected to separate congested lines are also congested due to KVL and voltage angle limitations)
        raw_flow['Value'] = np.absolute(raw_flow['Value']).values
        organized_flow = raw_flow.pivot(index='Time', columns='Line', values='Value').reset_index(drop=True)
        for l_name in line_names:
            organized_flow[l_name] = organized_flow[l_name]/line_params.loc[line_params['line']==l_name, 'limit'].values[0]*100

        organized_flow_line_names = organized_flow.columns.values
        organized_flow_array = organized_flow.values

        for t in range(0, organized_flow_array.shape[0]):
            congested_lines = organized_flow_line_names[np.where(organized_flow_array[t,:]==100)[0]]
            congested_nodes = [a.split('_')[1] for a in congested_lines] + [b.split('_')[2] for b in congested_lines]
            congested_nodes = [*set(congested_nodes)]
       
            all_congested_lines = []
            for c in congested_nodes:
                all_congested_lines.extend([s for s in line_names if c in s])

            all_congested_lines = [*set(all_congested_lines)]
            all_congested_lines_idx = [np.where(organized_flow_line_names==q)[0][0] for q in all_congested_lines]

            organized_flow_array[t, all_congested_lines_idx] = 100

        organized_flow_df = pd.DataFrame(organized_flow_array, columns=organized_flow_line_names)
        WECC_hourly_line_utilization = organized_flow_df.mean(axis=1).values
        WECC_interregional_line_utilization = organized_flow_df.loc[:, Interregional_lines].mean(axis=1).values
        WECC_intraregional_line_utilization = organized_flow_df.loc[:, Intraregional_lines].mean(axis=1).values

        #Creating a dataframe and calculating percentiles
        globals()[f'Hourly_Datasets_{sc}_{year}'] = pd.DataFrame(zip(WECC_hourly_average_LMP, WECC_hourly_total_LOL, WECC_hourly_total_demand, WECC_hourly_total_renewables, WECC_hourly_SoC_availability, WECC_hourly_line_utilization, WECC_interregional_line_utilization, WECC_intraregional_line_utilization), 
                                                                 columns=['LMP','LOL','Demand','Renewables','SoC','Congestion','Interregional Congestion','Intraregional Congestion'], index=hourly_timestamp)
        
        #Calculating percentiles of the relevant data
        globals()[f'Hourly_Datasets_{sc}_{year}']['Demand Percentile'] = globals()[f'Hourly_Datasets_{sc}_{year}']['Demand'].rank(method='average', pct=True)*100
        globals()[f'Hourly_Datasets_{sc}_{year}']['Renewables Percentile'] = globals()[f'Hourly_Datasets_{sc}_{year}']['Renewables'].rank(method='average', pct=True)*100
        globals()[f'Hourly_Datasets_{sc}_{year}']['SoC Percentile'] = globals()[f'Hourly_Datasets_{sc}_{year}']['SoC'].rank(method='average', pct=True)*100
        globals()[f'Hourly_Datasets_{sc}_{year}']['Congestion Percentile'] = globals()[f'Hourly_Datasets_{sc}_{year}']['Congestion'].rank(method='average', pct=True)*100

        #Adding day of year and hour of day columns
        globals()[f'Hourly_Datasets_{sc}_{year}']['Day of Year'] = globals()[f'Hourly_Datasets_{sc}_{year}'].index.dayofyear
        globals()[f'Hourly_Datasets_{sc}_{year}']['Hour of Day'] = globals()[f'Hourly_Datasets_{sc}_{year}'].index.hour

        globals()[f'Hourly_Datasets_{sc}_{year}']['Day of Year Percentile'] = globals()[f'Hourly_Datasets_{sc}_{year}']['Day of Year'].rank(method='average', pct=True)*100
        globals()[f'Hourly_Datasets_{sc}_{year}']['Hour of Day Percentile'] = globals()[f'Hourly_Datasets_{sc}_{year}']['Hour of Day'].rank(method='average', pct=True)*100

        globals()[f'Hourly_Datasets_LMP_{sc}_{year}'] = globals()[f'Hourly_Datasets_{sc}_{year}'].loc[(globals()[f'Hourly_Datasets_{sc}_{year}']['LMP']>=globals()[f'Hourly_Datasets_{sc}_{year}']['LMP'].quantile(quantile)) & (globals()[f'Hourly_Datasets_{sc}_{year}']['LOL']==0)]
        globals()[f'Hourly_Datasets_LMP_{sc}_{year}'].reset_index(inplace=True, drop=True)

        globals()[f'Hourly_Datasets_LOL_{sc}_{year}'] = globals()[f'Hourly_Datasets_{sc}_{year}'].loc[globals()[f'Hourly_Datasets_{sc}_{year}']['LOL']>0]
        globals()[f'Hourly_Datasets_LOL_{sc}_{year}'].reset_index(inplace=True, drop=True)

        if year == years[0]:
            globals()[f'All_LMP_Filtered_{sc}'] = globals()[f'Hourly_Datasets_LMP_{sc}_{year}']
            globals()[f'All_LOL_Filtered_{sc}'] = globals()[f'Hourly_Datasets_LOL_{sc}_{year}']

        else:
            globals()[f'All_LMP_Filtered_{sc}'] = pd.concat([globals()[f'All_LMP_Filtered_{sc}'], globals()[f'Hourly_Datasets_LMP_{sc}_{year}']], ignore_index=True)
            globals()[f'All_LOL_Filtered_{sc}'] = pd.concat([globals()[f'All_LOL_Filtered_{sc}'], globals()[f'Hourly_Datasets_LOL_{sc}_{year}']], ignore_index=True)


#Organizing data for plotting violinplots
for idx, sc in enumerate(scenarios):

    percentile_list = []
    grid_stress_type = []
    metric_type = []

    #Adding LMP based data to respective lists
    percentile_list.extend(globals()[f'All_LMP_Filtered_{sc}']['Demand Percentile'].values)
    grid_stress_type.extend(np.repeat('LMP', len(globals()[f'All_LMP_Filtered_{sc}'])))
    metric_type.extend(np.repeat('Demand', len(globals()[f'All_LMP_Filtered_{sc}'])))

    percentile_list.extend(globals()[f'All_LMP_Filtered_{sc}']['Renewables'].values)
    grid_stress_type.extend(np.repeat('LMP', len(globals()[f'All_LMP_Filtered_{sc}'])))
    metric_type.extend(np.repeat('Renewables', len(globals()[f'All_LMP_Filtered_{sc}'])))

    percentile_list.extend(globals()[f'All_LMP_Filtered_{sc}']['SoC'].values)
    grid_stress_type.extend(np.repeat('LMP', len(globals()[f'All_LMP_Filtered_{sc}'])))
    metric_type.extend(np.repeat('SoC', len(globals()[f'All_LMP_Filtered_{sc}'])))

    percentile_list.extend(globals()[f'All_LMP_Filtered_{sc}']['Interregional Congestion'].values)
    grid_stress_type.extend(np.repeat('LMP', len(globals()[f'All_LMP_Filtered_{sc}'])))
    metric_type.extend(np.repeat('Interregional Congestion', len(globals()[f'All_LMP_Filtered_{sc}'])))

    percentile_list.extend(globals()[f'All_LMP_Filtered_{sc}']['Intraregional Congestion'].values)
    grid_stress_type.extend(np.repeat('LMP', len(globals()[f'All_LMP_Filtered_{sc}'])))
    metric_type.extend(np.repeat('Intraregional Congestion', len(globals()[f'All_LMP_Filtered_{sc}'])))

    percentile_list.extend(globals()[f'All_LMP_Filtered_{sc}']['Day of Year'].values)
    grid_stress_type.extend(np.repeat('LMP', len(globals()[f'All_LMP_Filtered_{sc}'])))
    metric_type.extend(np.repeat('Day of Year', len(globals()[f'All_LMP_Filtered_{sc}'])))

    percentile_list.extend(globals()[f'All_LMP_Filtered_{sc}']['Hour of Day'].values)
    grid_stress_type.extend(np.repeat('LMP', len(globals()[f'All_LMP_Filtered_{sc}'])))
    metric_type.extend(np.repeat('Hour of Day', len(globals()[f'All_LMP_Filtered_{sc}'])))


    #Adding LOL based data to respective lists
    percentile_list.extend(globals()[f'All_LOL_Filtered_{sc}']['Demand Percentile'].values)
    grid_stress_type.extend(np.repeat('LOL', len(globals()[f'All_LOL_Filtered_{sc}'])))
    metric_type.extend(np.repeat('Demand', len(globals()[f'All_LOL_Filtered_{sc}'])))

    percentile_list.extend(globals()[f'All_LOL_Filtered_{sc}']['Renewables'].values)
    grid_stress_type.extend(np.repeat('LOL', len(globals()[f'All_LOL_Filtered_{sc}'])))
    metric_type.extend(np.repeat('Renewables', len(globals()[f'All_LOL_Filtered_{sc}'])))

    percentile_list.extend(globals()[f'All_LOL_Filtered_{sc}']['SoC'].values)
    grid_stress_type.extend(np.repeat('LOL', len(globals()[f'All_LOL_Filtered_{sc}'])))
    metric_type.extend(np.repeat('SoC', len(globals()[f'All_LOL_Filtered_{sc}'])))

    percentile_list.extend(globals()[f'All_LOL_Filtered_{sc}']['Interregional Congestion'].values)
    grid_stress_type.extend(np.repeat('LOL', len(globals()[f'All_LOL_Filtered_{sc}'])))
    metric_type.extend(np.repeat('Interregional Congestion', len(globals()[f'All_LOL_Filtered_{sc}'])))

    percentile_list.extend(globals()[f'All_LOL_Filtered_{sc}']['Intraregional Congestion'].values)
    grid_stress_type.extend(np.repeat('LOL', len(globals()[f'All_LOL_Filtered_{sc}'])))
    metric_type.extend(np.repeat('Intraregional Congestion', len(globals()[f'All_LOL_Filtered_{sc}'])))

    percentile_list.extend(globals()[f'All_LOL_Filtered_{sc}']['Day of Year'].values)
    grid_stress_type.extend(np.repeat('LOL', len(globals()[f'All_LOL_Filtered_{sc}'])))
    metric_type.extend(np.repeat('Day of Year', len(globals()[f'All_LOL_Filtered_{sc}'])))

    percentile_list.extend(globals()[f'All_LOL_Filtered_{sc}']['Hour of Day'].values)
    grid_stress_type.extend(np.repeat('LOL', len(globals()[f'All_LOL_Filtered_{sc}'])))
    metric_type.extend(np.repeat('Hour of Day', len(globals()[f'All_LOL_Filtered_{sc}'])))

    #Creating the final dataframe
    globals()[f'Plotting_Data_{sc}'] = pd.DataFrame(zip(percentile_list, metric_type, grid_stress_type,), columns=['Percentile', 'Metric', 'Type'])
    
    if idx == 0:
        Grid_stress_data = globals()[f'Plotting_Data_{sc}']
    else:
        Grid_stress_data = pd.concat([Grid_stress_data, globals()[f'Plotting_Data_{sc}']], ignore_index=True)



############## Plotting metric distributions to understand reasons for grid stress (all scenarios separately) ##############

# Initializing the figure

# Creating the outer 4 panel layout of the figure
plt.rcParams['font.sans-serif'] = "Arial"

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
inner_axis_width_1 = 5.6
inner_axis_height_1 = 3.9
inner_axis_width_2 = 1
inner_axis_height_2 = 3.9
inner_axis_width_3 = 1
inner_axis_height_3 = 3.9

ax1_in1 = inset_axes(ax1, width=inner_axis_width_1, height=inner_axis_height_1, loc='center left', borderpad=1.5, bbox_to_anchor=(0.03, 0.53), bbox_transform=ax1.transAxes) 
ax1_in2 = inset_axes(ax1, width=inner_axis_width_2, height=inner_axis_height_2, loc='center left', borderpad=1.5, bbox_to_anchor=(0.693, 0.53), bbox_transform=ax1.transAxes)
ax1_in3 = inset_axes(ax1, width=inner_axis_width_3, height=inner_axis_height_3, loc='center left', borderpad=1.5, bbox_to_anchor=(0.853, 0.53), bbox_transform=ax1.transAxes)

ax2_in1 = inset_axes(ax2, width=inner_axis_width_1, height=inner_axis_height_1, loc='center left', borderpad=1.5, bbox_to_anchor=(0.03, 0.53), bbox_transform=ax2.transAxes)
ax2_in2 = inset_axes(ax2, width=inner_axis_width_2, height=inner_axis_height_2, loc='center left', borderpad=1.5, bbox_to_anchor=(0.693, 0.53), bbox_transform=ax2.transAxes)
ax2_in3 = inset_axes(ax2, width=inner_axis_width_3, height=inner_axis_height_3, loc='center left', borderpad=1.5, bbox_to_anchor=(0.853, 0.53), bbox_transform=ax2.transAxes)

ax3_in1 = inset_axes(ax3, width=inner_axis_width_1, height=inner_axis_height_1, loc='center left', borderpad=1.5, bbox_to_anchor=(0.03, 0.53), bbox_transform=ax3.transAxes)
ax3_in2 = inset_axes(ax3, width=inner_axis_width_2, height=inner_axis_height_2, loc='center left', borderpad=1.5, bbox_to_anchor=(0.693, 0.53), bbox_transform=ax3.transAxes)
ax3_in3 = inset_axes(ax3, width=inner_axis_width_3, height=inner_axis_height_3, loc='center left', borderpad=1.5, bbox_to_anchor=(0.853, 0.53), bbox_transform=ax3.transAxes)

ax4_in1 = inset_axes(ax4, width=inner_axis_width_1, height=inner_axis_height_1, loc='center left', borderpad=1.5, bbox_to_anchor=(0.03, 0.53), bbox_transform=ax4.transAxes)
ax4_in2 = inset_axes(ax4, width=inner_axis_width_2, height=inner_axis_height_2, loc='center left', borderpad=1.5, bbox_to_anchor=(0.693, 0.53), bbox_transform=ax4.transAxes)
ax4_in3 = inset_axes(ax4, width=inner_axis_width_3, height=inner_axis_height_3, loc='center left', borderpad=1.5, bbox_to_anchor=(0.853, 0.53), bbox_transform=ax4.transAxes)

############## Plotting grid stress distributions ##############
for idx, sc in enumerate(scenarios):

    sns.violinplot(data=globals()[f'Plotting_Data_{sc}'].loc[~globals()[f'Plotting_Data_{sc}']['Metric'].isin(['Day of Year','Hour of Day'])], x='Metric', y='Percentile', hue='Type', inner="box", split=True, ax=globals()[f'ax{idx+1}_in1'], fill=True, 
               palette=['#FFDBC0','#B8E6FF'], cut=0.2, gap=0.08, density_norm='width', legend=False, inner_kws={'box_width':8, 'whis_width':2, 'color':'#000000'},
               order=['Demand','Renewables','SoC','Intraregional Congestion','Interregional Congestion'])

    globals()[f'ax{idx+1}_in1'].set_xlabel('')
    globals()[f'ax{idx+1}_in1'].set_ylabel('')
    globals()[f'ax{idx+1}_in1'].set_xticks(['Demand','Renewables','SoC','Intraregional Congestion','Interregional Congestion'], 
              ['Demand\nPercentile','Available\nRenewable\nPercentage','Available\nStorage\nPercentage','Intraregional\nLine Use\nPercentage','Interregional\nLine Use\nPercentage'], fontsize=10.5)

    globals()[f'ax{idx+1}_in1'].set_yticks([0,10,20,30,40,50,60,70,80,90,100], ['0%','10%','20%','30%','40%','50%','60%','70%','80%','90%','100%'], fontsize=10.5)
    globals()[f'ax{idx+1}_in1'].set_ylim([0,100])
    globals()[f'ax{idx+1}_in1'].grid(axis='both', color='#949494', alpha=0.3)
    globals()[f'ax{idx+1}_in1'].tick_params(axis='both', color='#FFFFFF')

    handles = []
    patch1 = Patch(facecolor='#FFDBC0', edgecolor='#000000',label='High LMP',linewidth=0.5)
    patch2 = Patch(facecolor='#B8E6FF', edgecolor='#000000',label='Unserved\nEnergy',linewidth=0.5)
    handles.extend([patch1,patch2])
    globals()[f'ax{idx+1}_in1'].legend(handles=handles,loc='upper center', ncol=1, fontsize=10.5, frameon=True, framealpha=1)

    sns.violinplot(data=globals()[f'Plotting_Data_{sc}'].loc[globals()[f'Plotting_Data_{sc}']['Metric']=='Day of Year'], x='Metric', y='Percentile', hue='Type', inner="box", split=True, ax=globals()[f'ax{idx+1}_in2'], fill=True, 
               palette=['#FFDBC0','#B8E6FF'], cut=0.2, gap=0.08, density_norm='width', legend=False, inner_kws={'box_width':8, 'whis_width':2, 'color':'#000000'})

    globals()[f'ax{idx+1}_in2'].set_xlabel('')
    globals()[f'ax{idx+1}_in2'].set_ylabel('')

    globals()[f'ax{idx+1}_in2'].set_xticks(['Day of Year'], ['Day of Year'], fontsize=10.5)
    globals()[f'ax{idx+1}_in2'].set_ylim([1,365])
    globals()[f'ax{idx+1}_in2'].set_yticks([16,46,74,105,135,166,196,227,258,288,319,349], ['Jan 15','Feb 15','Mar 15','Apr 15','May 15','Jun 15','Jul 15','Aug 15','Sep 15','Oct 15','Nov 15','Dec 15'], fontsize=10.5)
    globals()[f'ax{idx+1}_in2'].grid(axis='both', color='#949494', alpha=0.3)
    globals()[f'ax{idx+1}_in2'].tick_params(axis='both', color='#FFFFFF')

    sns.violinplot(data=globals()[f'Plotting_Data_{sc}'].loc[globals()[f'Plotting_Data_{sc}']['Metric']=='Hour of Day'], x='Metric', y='Percentile', hue='Type', inner="box", split=True, ax=globals()[f'ax{idx+1}_in3'], fill=True, 
               palette=['#FFDBC0','#B8E6FF'], cut=0.2, gap=0.08, density_norm='width', legend=False, inner_kws={'box_width':8, 'whis_width':2, 'color':'#000000'})

    globals()[f'ax{idx+1}_in3'].set_xlabel('')
    globals()[f'ax{idx+1}_in3'].set_ylabel('')

    globals()[f'ax{idx+1}_in3'].set_xticks(['Hour of Day'], ['Hour of Day'], fontsize=10.5)
    globals()[f'ax{idx+1}_in3'].set_ylim([0,23])
    globals()[f'ax{idx+1}_in3'].set_yticks([0,2,4,6,8,10,12,14,16,18,20,22], ['12 AM','2 AM','4 AM','6 AM','8 AM','10 AM','12 PM','2 PM','4 PM','6 PM','8 PM','10 PM'], fontsize=10.5)
    globals()[f'ax{idx+1}_in3'].grid(axis='both', color='#949494', alpha=0.3)
    globals()[f'ax{idx+1}_in3'].tick_params(axis='both', color='#FFFFFF')

    globals()[f'ax{idx+1}_in1'].tick_params(axis='x', which='both', pad=-2)
    globals()[f'ax{idx+1}_in1'].tick_params(axis='y', which='both', pad=0.01)
    globals()[f'ax{idx+1}_in2'].tick_params(axis='both', which='both', pad=0.01)
    globals()[f'ax{idx+1}_in3'].tick_params(axis='both', which='both', pad=0.01)
    

plt.savefig(f'Grid_stress_reasons_{t_scenario}_scenarios_all_together.png', dpi=400, bbox_inches='tight')
plt.show()
plt.clf()

