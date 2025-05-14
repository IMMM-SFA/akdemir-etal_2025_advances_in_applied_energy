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
scenarios = [f'rcp45{t_scenario}_ssp5', f'rcp85{t_scenario}_ssp5', f'rcp45{t_scenario}_ssp3', f'rcp85{t_scenario}_ssp3']
##########################################

#Reading nodal topology
all_nodes = pd.read_csv('../../Supplementary_Data/BA_Topology_Files/selected_nodes_125.csv', header=0)
all_node_numbers = [*all_nodes['SelectedNodes']]
all_node_strings = [f'bus_{i}' for i in all_node_numbers]
topology_10k_nodes = pd.read_csv('../../Supplementary_Data/BA_Topology_Files/10k_Load.csv', header=0)
nodal_info = pd.read_csv('../../Supplementary_Data/BA_Topology_Files/Nodal_information.csv', header=0)

hourly_timestamp = pd.date_range(start='2015-01-01 00:00:00', end='2015-12-31 23:00:00', freq='h')
daily_timestamp = pd.date_range(start='2015-01-01', end='2015-12-31', freq='d')

#Reading LMP and demand and calculating demand weighted average LMP timeseries and total demand timeseries
for sc in scenarios:
    for year in years:

        for l in glob.glob(f'Z:/im3/exp_b/exp_b_multi_model_coupling_west/experiment_runs/{run_name}/{sc}/go/output/native_output/{year}/duals/duals_{year}PI*.parquet'):
            raw_LMP = pd.read_parquet(l)

        for d in glob.glob(f'Z:/im3/exp_b/exp_b_multi_model_coupling_west/experiment_runs/{run_name}/{sc}/go/input/nodal_load_csv/{year}/nodal_load_*.csv'):
            demand = pd.read_csv(d, header=0)

        nodal_demand_weights = demand.sum()/demand.sum().sum()

        WECC_hourly_average_LMP = np.zeros(len(hourly_timestamp))
        for bus in all_node_strings:
            WECC_hourly_average_LMP = WECC_hourly_average_LMP + (raw_LMP.loc[raw_LMP['Bus']==bus]['Value'].values*nodal_demand_weights[bus])

        globals()[f'Hourly_LMP_{sc}_{year}'] = pd.DataFrame(WECC_hourly_average_LMP, columns=['LMP'], index=hourly_timestamp)
        globals()[f'Daily_LMP_{sc}_{year}'] = globals()[f'Hourly_LMP_{sc}_{year}'].resample('D').mean()
        globals()[f'Hourly_LMP_{sc}_{year}'].reset_index(drop=True, inplace=True)
        globals()[f'Daily_LMP_{sc}_{year}'].reset_index(drop=True, inplace=True)

        globals()[f'Hourly_Load_{sc}_{year}'] = pd.DataFrame(demand.sum(axis=1).values/1000, columns=['Load'], index=hourly_timestamp)
        globals()[f'Daily_Load_{sc}_{year}'] = globals()[f'Hourly_Load_{sc}_{year}'].resample('D').mean()
        globals()[f'Hourly_Load_{sc}_{year}'].reset_index(drop=True, inplace=True)
        globals()[f'Daily_Load_{sc}_{year}'].reset_index(drop=True, inplace=True)

for sc in scenarios:
    for year in years:

        if (sc == 'rcp45hotter_ssp5') or (sc == 'rcp45cooler_ssp5'):
            sc_leg = 'RCP 4.5/SSP 5'
        elif (sc == 'rcp85hotter_ssp5') or (sc == 'rcp85cooler_ssp5'):
            sc_leg = 'RCP 8.5/SSP 5'
        elif (sc == 'rcp45hotter_ssp3') or (sc == 'rcp45cooler_ssp3'):
            sc_leg = 'RCP 4.5/SSP 3'
        elif (sc == 'rcp85hotter_ssp3') or (sc == 'rcp85cooler_ssp3'):
            sc_leg = 'RCP 8.5/SSP 3'

        globals()[f'Hourly_LMP_{sc}_{year}']['Scenario'] = np.repeat(sc_leg, len(hourly_timestamp))
        globals()[f'Hourly_LMP_{sc}_{year}']['Year'] = np.repeat(year, len(hourly_timestamp))
        globals()[f'Daily_LMP_{sc}_{year}']['Scenario'] = np.repeat(sc_leg, len(daily_timestamp))
        globals()[f'Daily_LMP_{sc}_{year}']['Year'] = np.repeat(year, len(daily_timestamp))

        globals()[f'Hourly_Load_{sc}_{year}']['Scenario'] = np.repeat(sc_leg, len(hourly_timestamp))
        globals()[f'Hourly_Load_{sc}_{year}']['Year'] = np.repeat(year, len(hourly_timestamp))
        globals()[f'Daily_Load_{sc}_{year}']['Scenario'] = np.repeat(sc_leg, len(daily_timestamp))
        globals()[f'Daily_Load_{sc}_{year}']['Year'] = np.repeat(year, len(daily_timestamp))

        if (sc == scenarios[0]) and (year == years[0]):
            All_hourly_LMP = globals()[f'Hourly_LMP_{sc}_{year}']
            All_daily_LMP = globals()[f'Daily_LMP_{sc}_{year}']

            All_hourly_load = globals()[f'Hourly_Load_{sc}_{year}']
            All_daily_load = globals()[f'Daily_Load_{sc}_{year}']

        else:
            All_hourly_LMP = pd.concat([All_hourly_LMP, globals()[f'Hourly_LMP_{sc}_{year}']], ignore_index=True)
            All_daily_LMP = pd.concat([All_daily_LMP, globals()[f'Daily_LMP_{sc}_{year}']], ignore_index=True)

            All_hourly_load = pd.concat([All_hourly_load, globals()[f'Hourly_Load_{sc}_{year}']], ignore_index=True)
            All_daily_load = pd.concat([All_daily_load, globals()[f'Daily_Load_{sc}_{year}']], ignore_index=True)


############## Plotting LMP and LOL distributions ##############

plt.rcParams['font.sans-serif'] = "Arial"
plt.rcParams.update({'font.size': 16})
fig, ax = plt.subplots(3,1,figsize=(16,14), height_ratios=[0.4,1,1]) 

#Defining specifics of broken y axis
d = 0.8
kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12, linestyle="none", color='k', mec='k', mew=1, clip_on=False)

#Creating two subplots to have a divided y axis for LMP box plots
sns.boxplot(x='Year', y='LMP', data=All_daily_LMP, ax=ax[0], hue='Scenario', palette="coolwarm",order=years, showmeans=False,
                            meanprops={"marker": "s", "markerfacecolor": "black", "markeredgecolor": "black",'markersize':7})
ax[0].set_ylim(500,2000)
ax[0].spines['bottom'].set_visible(False)
ax[0].plot([1, 0], [0, 0], transform=ax[0].transAxes, **kwargs)
ax[0].set_ylabel('')
ax[0].set_xlabel('')
ax[0].set_xticks([])
ax[0].set_yticks([1000,1500,2000])
ax[0].legend(loc='upper left', ncol=1, fontsize=15)
sec1 = ax[0].secondary_yaxis('right')
sec1.set_yticks([1000,1500,2000])
sec1.set_yticklabels([])


sns.boxplot(x='Year', y='LMP', data=All_daily_LMP, ax=ax[1], hue='Scenario', palette="coolwarm",order=years, showmeans=False,
                            meanprops={"marker": "s", "markerfacecolor": "black", "markeredgecolor": "black",'markersize':7})
ax[1].set_ylim(0,90)
ax[1].spines['top'].set_visible(False)
ax[1].plot([0, 1], [1, 1], transform=ax[1].transAxes, **kwargs)
ax[1].set_ylabel('')
ax[1].set_xticklabels([])
ax[1].set_yticks([0,10,20,30,40,50,60,70,80])
ax[1].legend([],[], frameon=False)
sec2 = ax[1].secondary_yaxis('right')
sec2.set_yticks([0,10,20,30,40,50,60,70,80])
sec2.set_yticklabels([])
ax[1].set_xlabel('')
ax[1].text(-0.067,0.375,'Daily Average LMP ($/MWh)', weight='bold' ,transform=ax[1].transAxes, rotation=90, fontsize=17)

#Plotting demand box plots
sns.boxplot(x='Year', y='Load', data=All_daily_load, ax=ax[2], hue='Scenario', palette="coolwarm",order=years, showmeans=False,
                            meanprops={"marker": "s", "markerfacecolor": "black", "markeredgecolor": "black",'markersize':7})
kwargs2 = dict(marker=[(-1, -d), (1, d)], markersize=0, linestyle="none", color='k', mec='k', mew=1, clip_on=False)
ax[2].plot([1, 0], [0, 0], transform=ax[2].transAxes, **kwargs2)
ax[2].set_ylabel('Daily Average Demand (GWh)', weight='bold', fontsize=17)
ax[2].set_yticks([50,75,100,125,150,175,200,225,250])
sec3 = ax[2].secondary_yaxis('right')
sec3.set_yticks([50,75,100,125,150,175,200,225,250])
sec3.set_yticklabels([])
ax[2].set_xlabel('')
ax[2].legend(loc='upper left', ncol=1, fontsize=15)

fig.subplots_adjust(hspace=0.1)

plt.savefig(f'LMP_Demand_WECC_Boxplot_{t_scenario}_scenarios.png', dpi=400, bbox_inches='tight')
plt.show()
plt.clf()

        