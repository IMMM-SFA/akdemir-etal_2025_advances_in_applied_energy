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
import matplotlib as mpl
import matplotlib.ticker as mticker

##########################################
years = [2020, 2025, 2030, 2035, 2040, 2045, 2050, 2055]
t_scenario = 'hotter'
run_name = 'run_110824'
scenarios = [f'rcp45{t_scenario}_ssp3', f'rcp85{t_scenario}_ssp3', f'rcp45{t_scenario}_ssp5', f'rcp85{t_scenario}_ssp5']
percentile = 99
##########################################

#Reading nodal topology
all_nodes = pd.read_csv('../../Supplementary_Data/BA_Topology_Files/selected_nodes_125.csv', header=0)
all_node_numbers = [*all_nodes['SelectedNodes']]
all_node_strings = [f'bus_{i}' for i in all_node_numbers]
topology_10k_nodes = pd.read_csv('../../Supplementary_Data/BA_Topology_Files/10k_Load.csv', header=0)
nodal_info = pd.read_csv('../../Supplementary_Data/BA_Topology_Files/Nodal_information.csv', header=0)
BAs_df = pd.read_csv('../../Supplementary_Data/BA_Topology_Files/BAs.csv', header=0)
BAs_df.loc[5, 'Abbreviation'] = 'CAISO'
BAs_df.sort_values(by='Abbreviation', inplace=True)
nodes_to_BA = pd.read_csv('../../Supplementary_Data/BA_Topology_Files/nodes_to_BA_state.csv', header=0)

hourly_timestamp = pd.date_range(start='2015-01-01 00:00:00', end='2015-12-31 23:00:00', freq='h')
daily_timestamp = pd.date_range(start='2015-01-01', end='2015-12-31', freq='d')

#Finding nodes at each BA
BA_node_dict = {}

for b in range(len(BAs_df)):

    BA_name = BAs_df.loc[b, 'Name']
    BA_abb = BAs_df.loc[b, 'Abbreviation']

    all_nodes_BA = nodes_to_BA.loc[nodes_to_BA['NAME']==BA_name]['Number'].values
    intersecting_nodes = list(set(all_nodes_BA) & set(all_node_numbers))
    intersecting_nodes = [f'bus_{i}' for i in intersecting_nodes]

    BA_node_dict[BA_abb] = []
    BA_node_dict[BA_abb].extend(intersecting_nodes)

#Reading LMP and unserved energy results and calculating demand weighted average LMP timeseries, total unserved energy and total demand
for sc in scenarios:

    globals()[f'Yearly_Mean_LMP_{sc}'] = pd.DataFrame(np.zeros((len(years), len(BAs_df))), columns=BAs_df['Abbreviation'].values, index=years)
    globals()[f'Yearly_Max_LMP_{sc}'] = pd.DataFrame(np.zeros((len(years), len(BAs_df))), columns=BAs_df['Abbreviation'].values, index=years)

    globals()[f'Yearly_Median_LMP_{sc}'] = pd.DataFrame(np.zeros((len(years), len(BAs_df))), columns=BAs_df['Abbreviation'].values, index=years)
    globals()[f'Yearly_Deviation_LMP_{sc}'] = pd.DataFrame(np.zeros((len(years), len(BAs_df))), columns=BAs_df['Abbreviation'].values, index=years)
    globals()[f'Yearly_Percentile_LMP_{sc}'] = pd.DataFrame(np.zeros((len(years), len(BAs_df))), columns=BAs_df['Abbreviation'].values, index=years)

    globals()[f'Yearly_LOL_{sc}'] = pd.DataFrame(np.zeros((len(years), len(BAs_df))), columns=BAs_df['Abbreviation'].values, index=years)
    globals()[f'Yearly_Demand_{sc}'] = pd.DataFrame(np.zeros((len(years), len(BAs_df))), columns=BAs_df['Abbreviation'].values, index=years)


    for year in years:

        for l in glob.glob(f'Z:/im3/exp_b/exp_b_multi_model_coupling_west/experiment_runs/{run_name}/{sc}/go/output/native_output/{year}/duals/duals_{year}PI*.parquet'):
            raw_LMP = pd.read_parquet(l)

        for ll in glob.glob(f'Z:/im3/exp_b/exp_b_multi_model_coupling_west/experiment_runs/{run_name}/{sc}/go/output/native_output/{year}/slack/slack_{year}PI*.parquet'):
            raw_LOL = pd.read_parquet(ll)

        for d in glob.glob(f'Z:/im3/exp_b/exp_b_multi_model_coupling_west/experiment_runs/{run_name}/{sc}/go/input/nodal_load_csv/{year}/nodal_load_*.csv'):
            demand = pd.read_csv(d, header=0)

        for ba in BAs_df['Abbreviation']:
            
            #Calculating yearly total demand and saving it to relevant dataframe
            total_demand = demand.loc[:, BA_node_dict[ba]].sum().sum() #in MWh
            globals()[f'Yearly_Demand_{sc}'].loc[year, ba] = total_demand

            #Calculating yearly total unserved energy and saving it to relevant dataframe
            total_LOL = raw_LOL.loc[raw_LOL['Node'].isin(BA_node_dict[ba])]['Value'].sum() #in MWh
            globals()[f'Yearly_LOL_{sc}'].loc[year, ba] = total_LOL

            #Calculating yearly average LMP and saving it to relevant dataframe
            BA_LMP_timeseries = np.zeros(len(hourly_timestamp))

            for bus in BA_node_dict[ba]:
                demand_weight = demand.loc[:, bus].sum()/total_demand
                BA_LMP_timeseries = BA_LMP_timeseries + (raw_LMP.loc[raw_LMP['Bus']==bus]['Value'].values*demand_weight)
            
            globals()[f'Yearly_Mean_LMP_{sc}'].loc[year, ba] = np.mean(BA_LMP_timeseries)
            globals()[f'Yearly_Max_LMP_{sc}'].loc[year, ba] = np.max(BA_LMP_timeseries)
            globals()[f'Yearly_Median_LMP_{sc}'].loc[year, ba] = np.median(BA_LMP_timeseries)
            globals()[f'Yearly_Deviation_LMP_{sc}'].loc[year, ba] = np.std(BA_LMP_timeseries)
            globals()[f'Yearly_Percentile_LMP_{sc}'].loc[year, ba] = np.percentile(BA_LMP_timeseries, percentile)



############## Initializing the figure for mean LMP ##############

# Creating the outer 4 panel layout of the figure
plt.rcParams['font.sans-serif'] = "Arial"
plt.rcParams.update({'font.size': 10.5})

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
inner_axis_width_1 = 7.8
inner_axis_height_1 = "40%"
inner_axis_width_2 = 7.8
inner_axis_height_2 = "40%"

ax1_in1 = inset_axes(ax1, width=inner_axis_width_1, height=inner_axis_height_1, loc='upper left', borderpad=1.4, bbox_to_anchor=(0.03, 0.02, 0.9, 0.98), bbox_transform=ax1.transAxes)
ax1_in2 = inset_axes(ax1, width=inner_axis_width_2, height=inner_axis_height_2, loc='lower left', borderpad=1.4, bbox_to_anchor=(0.03, 0.11, 0.9, 0.98), bbox_transform=ax1.transAxes)

ax2_in1 = inset_axes(ax2, width=inner_axis_width_1, height=inner_axis_height_1, loc='upper left', borderpad=1.4, bbox_to_anchor=(0.03, 0.02, 0.9, 0.98), bbox_transform=ax2.transAxes)
ax2_in2 = inset_axes(ax2, width=inner_axis_width_2, height=inner_axis_height_2, loc='lower left', borderpad=1.4, bbox_to_anchor=(0.03, 0.11, 0.9, 0.98), bbox_transform=ax2.transAxes)

ax3_in1 = inset_axes(ax3, width=inner_axis_width_1, height=inner_axis_height_1, loc='upper left', borderpad=1.4, bbox_to_anchor=(0.03, 0.02, 0.9, 0.98), bbox_transform=ax3.transAxes)
ax3_in2 = inset_axes(ax3, width=inner_axis_width_2, height=inner_axis_height_2, loc='lower left', borderpad=1.4, bbox_to_anchor=(0.03, 0.11, 0.9, 0.98), bbox_transform=ax3.transAxes)

ax4_in1 = inset_axes(ax4, width=inner_axis_width_1, height=inner_axis_height_1, loc='upper left', borderpad=1.4, bbox_to_anchor=(0.03, 0.02, 0.9, 0.98), bbox_transform=ax4.transAxes)
ax4_in2 = inset_axes(ax4, width=inner_axis_width_2, height=inner_axis_height_2, loc='lower left', borderpad=1.4, bbox_to_anchor=(0.03, 0.11, 0.9, 0.98), bbox_transform=ax4.transAxes)

############## Plotting heatmaps ##############

norm1 = mpl.colors.BoundaryNorm(boundaries=[35,40,45,50,55,60,70,80], ncolors=256, extend='both')
cmap1 = 'YlOrRd'

norm2 = mpl.colors.BoundaryNorm(boundaries=[0,0.02,0.05,0.1,0.25,0.5,1], ncolors=256, extend='both')
cmap2 = 'Greys'

for idx, sc in enumerate(scenarios):

    sns.heatmap(globals()[f'Yearly_Mean_LMP_{sc}'], ax=globals()[f'ax{idx+1}_in1'], cmap=cmap1, cbar=True, linewidths=.5 ,cbar_kws={'shrink':0.4, 'label':'Yearly Average LMP\n($/MWh)', 'anchor':(0.4,0.96)}, norm=norm1)
    globals()[f'ax{idx+1}_in1'].set_xticklabels([])

    sns.heatmap(globals()[f'Yearly_LOL_{sc}']/globals()[f'Yearly_Demand_{sc}']*100, ax=globals()[f'ax{idx+1}_in2'], cmap=cmap2, cbar=True, linewidths=.5 ,cbar_kws={'shrink':0.4, 'label':'Yearly Unserved Energy\nto Demand Ratio (%)', 'anchor':(0.4,0.21),'ticks':[0,0.02,0.05,0.1,0.25,0.5,1],'format':mticker.FixedFormatter(['0','0.02','0.05','0.1','0.25','0.5','1'])}, norm=norm2)

plt.savefig(f'Mean_LMP_heatmaps_{t_scenario}_scenarios.png', dpi=500, bbox_inches='tight')
plt.show()
plt.clf()




############## Initializing the figure for max LMP ##############

# Creating the outer 4 panel layout of the figure
plt.rcParams['font.sans-serif'] = "Arial"
plt.rcParams.update({'font.size': 10.5})

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
inner_axis_width_1 = 7.8
inner_axis_height_1 = "40%"
inner_axis_width_2 = 7.8
inner_axis_height_2 = "40%"

ax1_in1 = inset_axes(ax1, width=inner_axis_width_1, height=inner_axis_height_1, loc='upper left', borderpad=1.4, bbox_to_anchor=(0.03, 0.02, 0.9, 0.98), bbox_transform=ax1.transAxes)
ax1_in2 = inset_axes(ax1, width=inner_axis_width_2, height=inner_axis_height_2, loc='lower left', borderpad=1.4, bbox_to_anchor=(0.03, 0.11, 0.9, 0.98), bbox_transform=ax1.transAxes)

ax2_in1 = inset_axes(ax2, width=inner_axis_width_1, height=inner_axis_height_1, loc='upper left', borderpad=1.4, bbox_to_anchor=(0.03, 0.02, 0.9, 0.98), bbox_transform=ax2.transAxes)
ax2_in2 = inset_axes(ax2, width=inner_axis_width_2, height=inner_axis_height_2, loc='lower left', borderpad=1.4, bbox_to_anchor=(0.03, 0.11, 0.9, 0.98), bbox_transform=ax2.transAxes)

ax3_in1 = inset_axes(ax3, width=inner_axis_width_1, height=inner_axis_height_1, loc='upper left', borderpad=1.4, bbox_to_anchor=(0.03, 0.02, 0.9, 0.98), bbox_transform=ax3.transAxes)
ax3_in2 = inset_axes(ax3, width=inner_axis_width_2, height=inner_axis_height_2, loc='lower left', borderpad=1.4, bbox_to_anchor=(0.03, 0.11, 0.9, 0.98), bbox_transform=ax3.transAxes)

ax4_in1 = inset_axes(ax4, width=inner_axis_width_1, height=inner_axis_height_1, loc='upper left', borderpad=1.4, bbox_to_anchor=(0.03, 0.02, 0.9, 0.98), bbox_transform=ax4.transAxes)
ax4_in2 = inset_axes(ax4, width=inner_axis_width_2, height=inner_axis_height_2, loc='lower left', borderpad=1.4, bbox_to_anchor=(0.03, 0.11, 0.9, 0.98), bbox_transform=ax4.transAxes)

############## Plotting heatmaps ##############

norm1 = mpl.colors.BoundaryNorm(boundaries=[50,60,70,80,90,100,500,1000], ncolors=256, extend='both')
cmap1 = 'YlOrRd'

norm2 = mpl.colors.BoundaryNorm(boundaries=[0,0.02,0.05,0.1,0.25,0.5,1], ncolors=256, extend='both')
cmap2 = 'Greys'

for idx, sc in enumerate(scenarios):

    sns.heatmap(globals()[f'Yearly_Max_LMP_{sc}'], ax=globals()[f'ax{idx+1}_in1'], cmap=cmap1, cbar=True, linewidths=.5 ,cbar_kws={'shrink':0.4, 'label':'Yearly Maximum LMP ($/MWh)', 'anchor':(0.4,0.96)}, norm=norm1)
    globals()[f'ax{idx+1}_in1'].set_xticklabels([])

    sns.heatmap(globals()[f'Yearly_LOL_{sc}']/globals()[f'Yearly_Demand_{sc}']*100, ax=globals()[f'ax{idx+1}_in2'], cmap=cmap2, cbar=True, linewidths=.5 ,cbar_kws={'shrink':0.4, 'label':'Yearly Unserved Energy\nto Demand Ratio (%)', 'anchor':(0.4,0.21),'ticks':[0,0.02,0.05,0.1,0.25,0.5,1],'format':mticker.FixedFormatter(['0','0.02','0.05','0.1','0.25','0.5','1'])}, norm=norm2)

plt.savefig(f'Max_LMP_heatmaps_{t_scenario}_scenarios.png', dpi=500, bbox_inches='tight')
plt.show()
plt.clf()




############## Initializing the figure for median LMP ##############

# Creating the outer 4 panel layout of the figure
plt.rcParams['font.sans-serif'] = "Arial"
plt.rcParams.update({'font.size': 10.5})

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
inner_axis_width_1 = 7.8
inner_axis_height_1 = "40%"
inner_axis_width_2 = 7.8
inner_axis_height_2 = "40%"

ax1_in1 = inset_axes(ax1, width=inner_axis_width_1, height=inner_axis_height_1, loc='upper left', borderpad=1.4, bbox_to_anchor=(0.03, 0.02, 0.9, 0.98), bbox_transform=ax1.transAxes)
ax1_in2 = inset_axes(ax1, width=inner_axis_width_2, height=inner_axis_height_2, loc='lower left', borderpad=1.4, bbox_to_anchor=(0.03, 0.11, 0.9, 0.98), bbox_transform=ax1.transAxes)

ax2_in1 = inset_axes(ax2, width=inner_axis_width_1, height=inner_axis_height_1, loc='upper left', borderpad=1.4, bbox_to_anchor=(0.03, 0.02, 0.9, 0.98), bbox_transform=ax2.transAxes)
ax2_in2 = inset_axes(ax2, width=inner_axis_width_2, height=inner_axis_height_2, loc='lower left', borderpad=1.4, bbox_to_anchor=(0.03, 0.11, 0.9, 0.98), bbox_transform=ax2.transAxes)

ax3_in1 = inset_axes(ax3, width=inner_axis_width_1, height=inner_axis_height_1, loc='upper left', borderpad=1.4, bbox_to_anchor=(0.03, 0.02, 0.9, 0.98), bbox_transform=ax3.transAxes)
ax3_in2 = inset_axes(ax3, width=inner_axis_width_2, height=inner_axis_height_2, loc='lower left', borderpad=1.4, bbox_to_anchor=(0.03, 0.11, 0.9, 0.98), bbox_transform=ax3.transAxes)

ax4_in1 = inset_axes(ax4, width=inner_axis_width_1, height=inner_axis_height_1, loc='upper left', borderpad=1.4, bbox_to_anchor=(0.03, 0.02, 0.9, 0.98), bbox_transform=ax4.transAxes)
ax4_in2 = inset_axes(ax4, width=inner_axis_width_2, height=inner_axis_height_2, loc='lower left', borderpad=1.4, bbox_to_anchor=(0.03, 0.11, 0.9, 0.98), bbox_transform=ax4.transAxes)

############## Plotting heatmaps ##############

norm1 = mpl.colors.BoundaryNorm(boundaries=[35,40,45,50,55,60,65,70], ncolors=256, extend='both')
cmap1 = 'YlOrRd'

norm2 = mpl.colors.BoundaryNorm(boundaries=[0,0.02,0.05,0.1,0.25,0.5,1], ncolors=256, extend='both')
cmap2 = 'Greys'

for idx, sc in enumerate(scenarios):

    sns.heatmap(globals()[f'Yearly_Median_LMP_{sc}'], ax=globals()[f'ax{idx+1}_in1'], cmap=cmap1, cbar=True, linewidths=.5 ,cbar_kws={'shrink':0.4, 'label':'Yearly Median LMP\n($/MWh)', 'anchor':(0.4,0.96)}, norm=norm1)
    globals()[f'ax{idx+1}_in1'].set_xticklabels([])

    sns.heatmap(globals()[f'Yearly_LOL_{sc}']/globals()[f'Yearly_Demand_{sc}']*100, ax=globals()[f'ax{idx+1}_in2'], cmap=cmap2, cbar=True, linewidths=.5 ,cbar_kws={'shrink':0.4, 'label':'Yearly Unserved Energy\nto Demand Ratio (%)', 'anchor':(0.4,0.21),'ticks':[0,0.02,0.05,0.1,0.25,0.5,1],'format':mticker.FixedFormatter(['0','0.02','0.05','0.1','0.25','0.5','1'])}, norm=norm2)

plt.savefig(f'Median_LMP_heatmaps_{t_scenario}_scenarios.png', dpi=500, bbox_inches='tight')
plt.show()
plt.clf()




############## Initializing the figure for standard deviation of LMP ##############

# Creating the outer 4 panel layout of the figure
plt.rcParams['font.sans-serif'] = "Arial"
plt.rcParams.update({'font.size': 10.5})

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
inner_axis_width_1 = 7.8
inner_axis_height_1 = "40%"
inner_axis_width_2 = 7.8
inner_axis_height_2 = "40%"

ax1_in1 = inset_axes(ax1, width=inner_axis_width_1, height=inner_axis_height_1, loc='upper left', borderpad=1.4, bbox_to_anchor=(0.03, 0.02, 0.9, 0.98), bbox_transform=ax1.transAxes)
ax1_in2 = inset_axes(ax1, width=inner_axis_width_2, height=inner_axis_height_2, loc='lower left', borderpad=1.4, bbox_to_anchor=(0.03, 0.11, 0.9, 0.98), bbox_transform=ax1.transAxes)

ax2_in1 = inset_axes(ax2, width=inner_axis_width_1, height=inner_axis_height_1, loc='upper left', borderpad=1.4, bbox_to_anchor=(0.03, 0.02, 0.9, 0.98), bbox_transform=ax2.transAxes)
ax2_in2 = inset_axes(ax2, width=inner_axis_width_2, height=inner_axis_height_2, loc='lower left', borderpad=1.4, bbox_to_anchor=(0.03, 0.11, 0.9, 0.98), bbox_transform=ax2.transAxes)

ax3_in1 = inset_axes(ax3, width=inner_axis_width_1, height=inner_axis_height_1, loc='upper left', borderpad=1.4, bbox_to_anchor=(0.03, 0.02, 0.9, 0.98), bbox_transform=ax3.transAxes)
ax3_in2 = inset_axes(ax3, width=inner_axis_width_2, height=inner_axis_height_2, loc='lower left', borderpad=1.4, bbox_to_anchor=(0.03, 0.11, 0.9, 0.98), bbox_transform=ax3.transAxes)

ax4_in1 = inset_axes(ax4, width=inner_axis_width_1, height=inner_axis_height_1, loc='upper left', borderpad=1.4, bbox_to_anchor=(0.03, 0.02, 0.9, 0.98), bbox_transform=ax4.transAxes)
ax4_in2 = inset_axes(ax4, width=inner_axis_width_2, height=inner_axis_height_2, loc='lower left', borderpad=1.4, bbox_to_anchor=(0.03, 0.11, 0.9, 0.98), bbox_transform=ax4.transAxes)

############## Plotting heatmaps ##############

norm1 = mpl.colors.BoundaryNorm(boundaries=[5,10,15,25,50,100,250], ncolors=256, extend='both')
cmap1 = 'YlOrRd'

norm2 = mpl.colors.BoundaryNorm(boundaries=[0,0.02,0.05,0.1,0.25,0.5,1], ncolors=256, extend='both')
cmap2 = 'Greys'

for idx, sc in enumerate(scenarios):

    sns.heatmap(globals()[f'Yearly_Deviation_LMP_{sc}'], ax=globals()[f'ax{idx+1}_in1'], cmap=cmap1, cbar=True, linewidths=.5 ,cbar_kws={'shrink':0.4, 'label':'Yearly Standard Deviation\nof LMP ($/MWh)', 'anchor':(0.4,0.96)}, norm=norm1)
    globals()[f'ax{idx+1}_in1'].set_xticklabels([])

    sns.heatmap(globals()[f'Yearly_LOL_{sc}']/globals()[f'Yearly_Demand_{sc}']*100, ax=globals()[f'ax{idx+1}_in2'], cmap=cmap2, cbar=True, linewidths=.5 ,cbar_kws={'shrink':0.4, 'label':'Yearly Unserved Energy\nto Demand Ratio (%)', 'anchor':(0.4,0.21),'ticks':[0,0.02,0.05,0.1,0.25,0.5,1],'format':mticker.FixedFormatter(['0','0.02','0.05','0.1','0.25','0.5','1'])}, norm=norm2)

plt.savefig(f'Deviation_LMP_heatmaps_{t_scenario}_scenarios.png', dpi=500, bbox_inches='tight')
plt.show()
plt.clf()




############## Initializing the figure for percentile LMP ##############

# Creating the outer 4 panel layout of the figure
plt.rcParams['font.sans-serif'] = "Arial"
plt.rcParams.update({'font.size': 10.5})

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
inner_axis_width_1 = 7.8
inner_axis_height_1 = "40%"
inner_axis_width_2 = 7.8
inner_axis_height_2 = "40%"

ax1_in1 = inset_axes(ax1, width=inner_axis_width_1, height=inner_axis_height_1, loc='upper left', borderpad=1.4, bbox_to_anchor=(0.03, 0.02, 0.9, 0.98), bbox_transform=ax1.transAxes)
ax1_in2 = inset_axes(ax1, width=inner_axis_width_2, height=inner_axis_height_2, loc='lower left', borderpad=1.4, bbox_to_anchor=(0.03, 0.11, 0.9, 0.98), bbox_transform=ax1.transAxes)

ax2_in1 = inset_axes(ax2, width=inner_axis_width_1, height=inner_axis_height_1, loc='upper left', borderpad=1.4, bbox_to_anchor=(0.03, 0.02, 0.9, 0.98), bbox_transform=ax2.transAxes)
ax2_in2 = inset_axes(ax2, width=inner_axis_width_2, height=inner_axis_height_2, loc='lower left', borderpad=1.4, bbox_to_anchor=(0.03, 0.11, 0.9, 0.98), bbox_transform=ax2.transAxes)

ax3_in1 = inset_axes(ax3, width=inner_axis_width_1, height=inner_axis_height_1, loc='upper left', borderpad=1.4, bbox_to_anchor=(0.03, 0.02, 0.9, 0.98), bbox_transform=ax3.transAxes)
ax3_in2 = inset_axes(ax3, width=inner_axis_width_2, height=inner_axis_height_2, loc='lower left', borderpad=1.4, bbox_to_anchor=(0.03, 0.11, 0.9, 0.98), bbox_transform=ax3.transAxes)

ax4_in1 = inset_axes(ax4, width=inner_axis_width_1, height=inner_axis_height_1, loc='upper left', borderpad=1.4, bbox_to_anchor=(0.03, 0.02, 0.9, 0.98), bbox_transform=ax4.transAxes)
ax4_in2 = inset_axes(ax4, width=inner_axis_width_2, height=inner_axis_height_2, loc='lower left', borderpad=1.4, bbox_to_anchor=(0.03, 0.11, 0.9, 0.98), bbox_transform=ax4.transAxes)

############## Plotting heatmaps ##############

norm1 = mpl.colors.BoundaryNorm(boundaries=[35,40,45,50,55,60,65,70], ncolors=256, extend='both')
cmap1 = 'YlOrRd'

norm2 = mpl.colors.BoundaryNorm(boundaries=[0,0.02,0.05,0.1,0.25,0.5,1], ncolors=256, extend='both')
cmap2 = 'Greys'

for idx, sc in enumerate(scenarios):

    sns.heatmap(globals()[f'Yearly_Percentile_LMP_{sc}'], ax=globals()[f'ax{idx+1}_in1'], cmap=cmap1, cbar=True, linewidths=.5 ,cbar_kws={'shrink':0.4, 'label':f'Yearly {percentile} Percentile\nof LMP ($/MWh)', 'anchor':(0.4,0.96)}, norm=norm1)
    globals()[f'ax{idx+1}_in1'].set_xticklabels([])

    sns.heatmap(globals()[f'Yearly_LOL_{sc}']/globals()[f'Yearly_Demand_{sc}']*100, ax=globals()[f'ax{idx+1}_in2'], cmap=cmap2, cbar=True, linewidths=.5 ,cbar_kws={'shrink':0.4, 'label':'Yearly Unserved Energy\nto Demand Ratio (%)', 'anchor':(0.4,0.21),'ticks':[0,0.02,0.05,0.1,0.25,0.5,1],'format':mticker.FixedFormatter(['0','0.02','0.05','0.1','0.25','0.5','1'])}, norm=norm2)

plt.savefig(f'Percentile_LMP_heatmaps_{t_scenario}_scenarios.png', dpi=500, bbox_inches='tight')
plt.show()
plt.clf()
