import pandas as pd
import numpy as np
import os
import yaml
from shutil import copy
import geopandas as gpd
from shapely.geometry import Point
from geopy.distance import geodesic
import sys

#Defining path to the config file
my_config_file_path = "TEP_config.yml"



def read_config_file(config_file: str) -> dict:
    with open(config_file, 'r') as f:
        return yaml.safe_load(f)



def prep_tep(config_file):
    
    config_dict = read_config_file(config_file)

    # ---- INPUT PATHS ----

    node_topology_10k_path = config_dict['node_topology_file']
    transmission_subregion_file_path = config_dict['transmission_subregion_file']
    
    gen_data_path = config_dict['gen_data_file']
    gen_mat_file_path = config_dict['gen_mat_file_file']
    fuel_price_path = config_dict['fuel_price_file']
    hydro_min_path = config_dict['hydro_min_file']   
    hydro_max_path = config_dict['hydro_max_file']
    load_input_path = config_dict['load_input_file']
    nuclear_input_path = config_dict['must_run_input_file']
    offshore_wind_input_path = config_dict['offshore_wind_input_file']
    onshore_wind_input_path = config_dict['onshore_wind_input_file']
    solar_input_path = config_dict['solar_input_file']
    line_to_bus_input_path = config_dict['line_to_bus_input_file']
    existing_line_input_path = config_dict['existing_line_param_file']

    # ---- OUTPUT PATHS ----

    gen_data_output_path = config_dict['gen_data_output_file']
    gen_mat_output_path = config_dict['gen_mat_output_file']
    fuel_price_output_path = config_dict['fuel_price_output_file']
    hydro_min_output_path = config_dict['hydro_min_output_file']
    hydro_max_output_path = config_dict['hydro_max_output_file']
    nuclear_output_path = config_dict['must_run_output_file']
    load_output_path = config_dict['load_output_file']
    offshore_wind_output_path = config_dict['offshore_wind_output_file']
    onshore_wind_output_path = config_dict['onshore_wind_output_file']
    solar_output_path = config_dict['solar_output_file']
    line_to_bus_output_path = config_dict['line_to_bus_output_file']
    existing_line_output_path = config_dict['existing_line_param_output_file']

    # ----- SETTINGS ------

    interregional_tep_penalty = int(config_dict['interregional_tep_penalty'])
    line_len_security_scalar = int(config_dict['line_len_security_scalar'])

    # --------------------

    #Defining the number of days in every month
    Days_in_months = [31,28,31,30,31,30,31,31,30,31,30,31]

    #Defining the number of hours in every month
    Hours_in_months = [m*24 for m in Days_in_months]
 
    #Reading and organizing generator data and copying to simulation folder
    gen_data = pd.read_csv(gen_data_path, header=0)
    gen_data_filt = gen_data.loc[:,["name","typ","node","maxcap","heat_rate","var_om"]].copy()
    gen_data_filt.to_csv(gen_data_output_path, index=False)

    #Copying generator/node matrix to simulation folder
    copy(gen_mat_file_path, gen_mat_output_path)

    #Reading and calculating average fuel price and hydropower for each month and copying to simulation folder
    fuel_price = pd.read_csv(fuel_price_path, header=0)
    fuel_price_monthly = pd.DataFrame(np.zeros((len(Days_in_months), fuel_price.shape[1])), columns=fuel_price.columns)

    hydro_min = pd.read_csv(hydro_min_path, header=0)
    hydro_min_monthly = pd.DataFrame(np.zeros((len(Days_in_months), hydro_min.shape[1])), columns=hydro_min.columns)

    hydro_max = pd.read_csv(hydro_max_path, header=0)
    hydro_max_monthly = pd.DataFrame(np.zeros((len(Days_in_months), hydro_max.shape[1])), columns=hydro_max.columns)

    day_counter = 0
    idx = 0
    for month_day in Days_in_months:

        if idx == 0:
            fuel_filt = fuel_price.loc[0:month_day-1,:].mean().values
            fuel_price_monthly.loc[idx,:] = fuel_filt

            hydro_min_filt = hydro_min.loc[0:month_day-1,:].mean().values
            hydro_min_monthly.loc[idx,:] = hydro_min_filt

            hydro_max_filt = hydro_max.loc[0:month_day-1,:].mean().values
            hydro_max_monthly.loc[idx,:] = hydro_max_filt

        else:
            fuel_filt = fuel_price.loc[day_counter:day_counter+month_day-1,:].mean().values
            fuel_price_monthly.loc[idx,:] = fuel_filt

            hydro_min_filt = hydro_min.loc[day_counter:day_counter+month_day-1,:].mean().values
            hydro_min_monthly.loc[idx,:] = hydro_min_filt

            hydro_max_filt = hydro_max.loc[day_counter:day_counter+month_day-1,:].mean().values
            hydro_max_monthly.loc[idx,:] = hydro_max_filt

        day_counter += month_day
        idx += 1


    fuel_price_monthly.to_csv(fuel_price_output_path, index=False)
    hydro_min_monthly.to_csv(hydro_min_output_path, index=False)
    hydro_max_monthly.to_csv(hydro_max_output_path, index=False)


    #Reading and organizing nuclear data and copying to simulation folder
    try:
        nuclear = pd.read_csv(nuclear_input_path, header=0)
    except:
        nuclear = pd.DataFrame()

    nuclear_monthly = pd.DataFrame(np.zeros((len(Days_in_months),hydro_min.shape[1])), columns=hydro_min.columns)

    for node in hydro_min.columns:

        try:
            nuclear_monthly.loc[:,node] = np.repeat(nuclear.loc[0,node],len(Days_in_months))

        except KeyError:
            nuclear_monthly.loc[:,node] = np.repeat(0,len(Days_in_months))

    nuclear_monthly.to_csv(nuclear_output_path, index=False)

    #Reading and calculating peak net demand, respective total demand, wind and solar for each month and copying to simulation folder
    load = pd.read_csv(load_input_path, header=0)
    load_monthly = pd.DataFrame(np.zeros((len(Days_in_months),load.shape[1])), columns=load.columns)

    offshore_wind = pd.read_csv(offshore_wind_input_path, header=0)
    offshore_wind_monthly = pd.DataFrame(np.zeros((len(Days_in_months),offshore_wind.shape[1])), columns=offshore_wind.columns)
    
    onshore_wind = pd.read_csv(onshore_wind_input_path, header=0)
    onshore_wind_monthly = pd.DataFrame(np.zeros((len(Days_in_months),onshore_wind.shape[1])), columns=onshore_wind.columns)

    solar = pd.read_csv(solar_input_path, header=0)
    solar_monthly = pd.DataFrame(np.zeros((len(Days_in_months),solar.shape[1])), columns=solar.columns)

    #Calculating nodal hourly net demand (i.e., demand - solar - wind - offshore wind)
    net_demand = load - solar - onshore_wind - offshore_wind
    net_demand[net_demand < 0] = 0
    net_demand_total = net_demand.sum(axis=1)

    hour_counter = 0
    idx = 0
    for month_hour in Hours_in_months:

        if idx == 0:
            loc_max_net_demand = net_demand_total.loc[0:month_hour-1].argmax()
            c_idx = net_demand_total.loc[0:month_hour-1].index[loc_max_net_demand]

            load_monthly.loc[idx,:] = load.loc[c_idx,:].values
            offshore_wind_monthly.loc[idx,:] = offshore_wind.loc[c_idx,:].values
            onshore_wind_monthly.loc[idx,:] = onshore_wind.loc[c_idx,:].values
            solar_monthly.loc[idx,:] = solar.loc[c_idx,:].values
        
        else:
            loc_max_net_demand = net_demand_total.loc[hour_counter:hour_counter+month_hour-1].argmax()
            c_idx = net_demand_total.loc[hour_counter:hour_counter+month_hour-1].index[loc_max_net_demand]  

            load_monthly.loc[idx,:] = load.loc[c_idx,:].values
            offshore_wind_monthly.loc[idx,:] = offshore_wind.loc[c_idx,:].values
            onshore_wind_monthly.loc[idx,:] = onshore_wind.loc[c_idx,:].values
            solar_monthly.loc[idx,:] = solar.loc[c_idx,:].values

        hour_counter += month_hour
        idx += 1


    load_monthly.to_csv(load_output_path, index=False)
    
    offshore_wind_monthly.to_csv(offshore_wind_output_path, index=False)
    
    onshore_wind_monthly.to_csv(onshore_wind_output_path, index=False)
    
    solar_monthly.to_csv(solar_output_path, index=False)

    #Copying line/node matrix to simulation folder
    copy(line_to_bus_input_path, line_to_bus_output_path)

    #Reading existing line parameters
    existing_lines = pd.read_csv(existing_line_input_path, header=0)

    #Reading node data
    all_node_IDs = [int(i[4:]) for i in load_monthly.columns]

    # Node path
    Node_dataset = pd.read_csv(node_topology_10k_path, header=0)
    Node_dataset = Node_dataset.loc[Node_dataset["Number"].isin(all_node_IDs)]
    Node_dataset.reset_index(inplace=True,drop=True)
    geometry = [Point(xy) for xy in zip(Node_dataset['Substation Longitude'],Node_dataset['Substation Latitude'])]
    nodes_gdf = gpd.GeoDataFrame(Node_dataset,crs='epsg:4326',geometry=geometry)
    nodes_gdf = nodes_gdf.to_crs("EPSG:3395")
    nodes_gdf['TPR'] = np.repeat('Unknown',len(nodes_gdf))

    #Reading and transmission region shapefile
    Transmission_gdf = gpd.read_file(transmission_subregion_file_path)
    Transmission_gdf = Transmission_gdf.to_crs("EPSG:3395")
    WECC_tr_gdf = Transmission_gdf.loc[Transmission_gdf['NAME']=='WESTERN ELECTRICITY COORDINATING COUNCIL (WECC)'].copy()
    WECC_tr_gdf.reset_index(inplace=True,drop=True)

    #Finding and saving transmision planning region of every node
    for NN in range(0,len(Node_dataset)):

        node_point = nodes_gdf.loc[NN,'geometry']
        TPR_idx = node_point.within(WECC_tr_gdf['geometry']).idxmax()
        TPR_name = WECC_tr_gdf.loc[TPR_idx,'SUBNAME']

        if TPR_name == 'CA-MX US':
            nodes_gdf.loc[NN,'TPR'] = 'CAISO'
        elif TPR_name == 'NWPP':
            nodes_gdf.loc[NN,'TPR'] = 'NorthernGrid'
        else:
            nodes_gdf.loc[NN,'TPR'] = 'WestConnect'

    #Creating columns for types, lengths and costs of transmission lines
    existing_lines['transmission_type'] = np.repeat('Unknown',len(existing_lines))
    existing_lines['length_mile'] = np.repeat(-1.5,len(existing_lines))
    existing_lines['inv_cost_$_per_MWmile'] = np.repeat(-1.5,len(existing_lines))

    #Iterating over transmission lines to calculate types, lengths and costs
    for LL in range(0,len(existing_lines)):

        line_name = existing_lines.loc[LL,'line']
        splitted_name = line_name.split('_')

        #Checking if the line crosses TPR borders or not
        first_TPR = nodes_gdf.loc[nodes_gdf['Number']==int(splitted_name[1])]['TPR'].values[0]
        second_TPR = nodes_gdf.loc[nodes_gdf['Number']==int(splitted_name[2])]['TPR'].values[0]

        if first_TPR == second_TPR:
            line_type = 'regional'
        else:
            line_type = 'interregional'

        existing_lines.loc[LL,'transmission_type'] = line_type

        #Calculating the length of line in miles
        first_coordinates = (nodes_gdf.loc[nodes_gdf['Number']==int(splitted_name[1])]['Substation Latitude'].values[0],
                            nodes_gdf.loc[nodes_gdf['Number']==int(splitted_name[1])]['Substation Longitude'].values[0])
        
        second_coordinates = (nodes_gdf.loc[nodes_gdf['Number']==int(splitted_name[2])]['Substation Latitude'].values[0],
                            nodes_gdf.loc[nodes_gdf['Number']==int(splitted_name[2])]['Substation Longitude'].values[0])
        
        #Adding line length adder to line lengths for security purposes
        line_length = geodesic(first_coordinates, second_coordinates).miles * (1+(line_len_security_scalar/100))
        existing_lines.loc[LL,'length_mile'] = round(line_length, 3)

        #Calculating $/MW-mile cost of each transmission line
        if line_length < 300: #Equation is y = 1.6875*x + 2671.25

            if line_type == 'regional':
                line_cost = (1.6875*line_length) + 2671.25
            else:
                line_cost = ((1.6875*line_length) + 2671.25) * (1+(interregional_tep_penalty/100))

            existing_lines.loc[LL,'inv_cost_$_per_MWmile'] = round(line_cost, 3)

        elif line_length >= 300 and line_length < 500: #Equation is y = -6.7*x + 5130
            
            if line_type == 'regional':
                line_cost = (-6.7*line_length) + 5130
            else:
                line_cost = ((-6.7*line_length) + 5130) * (1+(interregional_tep_penalty/100))

            existing_lines.loc[LL,'inv_cost_$_per_MWmile'] = round(line_cost, 3)

        elif line_length >= 500 and line_length < 1000: #Equation is y = -0.55*x + 2045

            if line_type == 'regional':
                line_cost = (-0.55*line_length) + 2045
            else:
                line_cost = ((-0.55*line_length) + 2045) * (1+(interregional_tep_penalty/100))
            
            existing_lines.loc[LL,'inv_cost_$_per_MWmile'] = round(line_cost, 3)

        else: #Equation is y = 1495

            if line_type == 'regional':
                line_cost = 1495 
            else:
                line_cost = 1495 * (1+(interregional_tep_penalty/100))

            existing_lines.loc[LL,'inv_cost_$_per_MWmile'] = round(line_cost, 3) 

    existing_lines.to_csv(existing_line_output_path, index=False)


    #Printing the status of the model setup script
    print('Model setup for transmission expansion model is finished.')

    return None



#Preparing TEP input database
prep_tep(my_config_file_path)
    