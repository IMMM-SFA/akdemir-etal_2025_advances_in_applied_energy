Zenodo DOI: ?

# akdemir-etal_2025_advances_in_applied_energy

**Grid Stress and Reliability in Future Electricity Grids: Impacts of Generation Mix, Weather, and Demand Growth**

Kerem Ziya Akdemir<sup>1\*</sup>, Kendall Mongird<sup>1</sup>, Cameron Bracken<sup>1</sup>, Casey D. Burleyson<sup>1</sup>, Jordan D. Kern<sup>2</sup>, Konstantinos Oikonomou<sup>1</sup>, Travis B. Thurber<sup>1</sup>, Chris R. Vernon<sup>1</sup>, Nathalie Voisin<sup>1,3</sup>, Mengqi Zhao<sup>1</sup>, and Jennie S. Rice<sup>1</sup>

<sup>1 </sup> Pacific Northwest National Laboratory, Richland, WA, USA  
<sup>2 </sup> North Carolina State University, Raleigh, NC, USA  
<sup>3 </sup> University of Washington, Seattle, WA, USA

\* corresponding author: keremziya.akdemir@pnnl.gov

## Abstract
The reliability power grids in the future will depend on how system planners account for the individual and joint uncertainties in demand growth from increased electrification and the surge in data centers, integration of renewable energy, and extreme weather events. This study introduces an open-source multisectoral, multiscale modeling framework that projects grid stress and reliability trends between 2020-2055 in the Western Interconnection of the United States. The framework integrates global to national energy-water-land dynamics with power plant siting and hourly grid operations modeling. We analyze future wholesale electricity price shocks and unserved energy events across eight scenarios spanning a wide but plausible range of greenhouse gas emissions constraints, generation mixes, extreme weather events, and socioeconomic changes. Our results show that future grids with a high percentage of renewable resources (primarily due to emissions constraints) have lower median wholesale electricity prices but more volatile prices as well as more frequent and severe unserved energy events compared to scenarios without emissions constraints which rely on more dispatchable generators. These events occur because of the higher proportion of solar and wind causes net demand curves to deepen during midday (i.e., duck curves get progressively severe), exacerbating the challenge of meeting electricity demand during evening peaks in summer months. Scenarios with higher population and economic growth are characterized by higher reliability and lower wholesale electricity prices than lower growth scenarios because of their reliance on dispatchable generators and lower fossil fuel extraction costs. Robust and co-optimized transmission and energy storage planning are recommended to maintain reliability in future electricity grids.

## Journal reference
To be updated with appropriate reference information once the paper is published.

## Code reference
To be updated with appropriate reference information once the meta-repository is published.

## Data reference

### Input data
| Dataset | Repository Link | DOI |
| --- | --- | --- |
| GO and TEP Inputs | https://data.msdlive.org/records/7art3-45280 | https://doi.org/10.57931/2497839 |

### Output data
| Dataset | Repository Link | DOI |
| --- | --- | --- |
| GO and TEP Outputs | https://data.msdlive.org/records/7art3-45280 | https://doi.org/10.57931/2497839 |
| CERF Outputs | https://data.msdlive.org/records/62fpt-0jr75 | https://doi.org/10.57931/2479527 |

### Supplementary data
All supplementary data can be found in the `supplementary_data` directory.

| Dataset | Description | Reference |
| --- | --- | --- |
| BA_Topology_Files/10k_Load.csv | Nodal information including number IDs, names, area names, voltages, angles, locations, and loads within 10000-nodal topology of the U.S. Western Interconnection | [ACTIVSg10k](https://electricgrids.engr.tamu.edu/electric-grid-test-cases/activsg10k/) |
| BA_Topology_Files/BAs | Names and abbreviations of 28 balancing authorities considered in the U.S. Western Interconnection | Created by authors |
| BA_Topology_Files/line_params_125.csv | Names, reactances and thermal limits of transmission lines within reduced 125-nodal topology of the U.S. Western Interconnection | Created by authors |
| BA_Topology_Files/Nodal_information.csv | Number IDs, names, area names, locations, transmission planning regions, and load weights of individual nodes within reduced 125-nodal topology of the U.S. Western Interconnection  | Created by authors |
| BA_Topology_Files/nodes_to_BA_state.csv | Nodal information including number IDs, names, area names, voltages, angles, locations, loads, geometries, balancing authority and state information within 10000-nodal topology of the U.S. Western Interconnection | [ACTIVSg10k](https://electricgrids.engr.tamu.edu/electric-grid-test-cases/activsg10k/) (Modified by authors) |
| BA_Topology_Files/selected_nodes_125.csv | Number IDs of the selected nodes within reduced 125-nodal topology of the U.S. Western Interconnection  | Created by authors |
| Shapefiles/NERC_regions | Folder including shapefile of North American Electric Reliability Corporation (NERC) regions | [HIFLD](https://hifld-geoplatform.hub.arcgis.com/maps/6b2af23c67f04f4cb01d88c61aaf558a) |
| Shapefiles/US_states | Folder including shapefile of U.S. census states | [U.S. EIA](https://atlas.eia.gov/maps/774019f31f8549c39b5c72f149bbe74e) |

## Contributing modeling software
| Model | Version | Model Repository Link | DOI of Specific Version |
| --- | --- | --- | --- |
| GO | v0.1.0 | https://github.com/IMMM-SFA/go | https://doi.org/10.5281/zenodo.15399795 |
| TEP | v1.1.0 | https://github.com/keremakdemir/Transmission_Expansion_Planner | https://doi.org/10.5281/zenodo.15413081 |
| GCAM-USA | v5.3 | https://github.com/JGCRI/gcam-core | https://doi.org/10.5281/zenodo.3908600 |
| CERF | v?.?.? | https://github.com/IMMM-SFA/cerf | ? |
| TELL | v1.1.0 | https://github.com/IMMM-SFA/tell | https://doi.org/10.5281/zenodo.8264217 |
| reV | v0.7.0 | https://github.com/NREL/reV | https://doi.org/10.5281/zenodo.7301491 |

## Reproduce my experiment
Use the scripts found in the `workflow` directory to reproduce the experiment presented in this publication. 
- Please check and make sure that all the necessary packages listed in `requirements.txt` are installed in your local Python environment.

### Steps
1. ...
2. ...
3. ...

Fill in detailed info here or link to other documentation to thoroughly walkthrough how to use the contents of this repository to reproduce your experiment. Below is an example.


1. Install the software components required to conduct the experiment from [contributing modeling software](#contributing-modeling-software)
2. Download and install the supporting [input data](#input-data) required to conduct the experiment
3. Run the following scripts in the `workflow` directory to re-create this experiment:

| Script Name | Description | How to Run |
| --- | --- | --- |
| `step_one.py` | Script to run the first part of my experiment | `python3 step_one.py -f /path/to/inputdata/file_one.csv` |
| `step_two.py` | Script to run the second part of my experiment | `python3 step_two.py -o /path/to/my/outputdir` |

4. Download and unzip the [output data](#output-data) from my experiment 
5. Run the following scripts in the `workflow` directory to compare my outputs to those from the publication

| Script Name | Description | How to Run |
| --- | --- | --- |
| `compare.py` | Script to compare my outputs to the original | `python3 compare.py --orig /path/to/original/data.csv --new /path/to/new/data.csv` |

## Reproduce my figures
Use the scripts found in the `figures` directory to reproduce the figures used in this publication. 
- Please check and make sure that all the necessary packages listed in `requirements.txt` are installed in your local Python environment.
- Please update all the paths in the scripts so that they point to the local paths of the downloaded [input](#input-data)/[output](#output-data)/[supplementary](#supplementary-data) datasets. 
- Setting `t_scenario = cooler` would produce the figures in the main body of the manuscript, whereas setting `t_scenario = hotter` would produce the figures in the supplementary information.

| Figure Number | Script/File Name | Description |
| --- | --- | --- |
| 1 | `Nodal_topology.py` | Plots the 125-nodal topology of GO model and three transmission planning regions of the U.S. Western Interconnection |
| 2 | `Experiment_flowchart.pptx` | Shows the flowchart of the experimental workflow to simulate grid stress and reliability |
| 3 | `Grid_futures.py` | Plots the changes in dispatchable generation capacity, renewable generation capacity, storage discharge capacity, intraregional transmission capacity, interregional transmission capacity, and average hourly demand between 2020-2055 |
| 4 | `Generation_mix.py` | Plots the annual generation mix in U.S. Western Interconnection between 2020-2055 |
| 5 | `LMP_demand_boxplots.py` | Plots the distributions of daily average LMP and daily average demand for the U.S. Western Interconnection between 2020-2055 |
| 6 | `LMP_LOL_heatmaps.py` | Plots the yearly average LMP and yearly unserved energy to demand ratio for each U.S. Western Interconnection balancing authority and for each simulation year between 2020-2055 |
| 7 | `Reasons_for_grid_stress.py` | Plots the demand, available renewable and storage capacities, intraregional and interregional transmission line usage, day of year and hour of day distributions of high LMP and unserved energy events considering all simulation years between 2020-2055 |
| 8 | `Storage_net_demand_trends.py` | Plots the average hour of day trends of storage capacity utilization and net demand for each simulation year between 2020-2055 |