import os
import pandas as pd
import numpy as np
import pyomo.environ as pyo
import sys
import yaml

#Defining path to the config file
my_config_file_path = "TEP_config.yml"

#Defining yearly transmission investment budget in USD
Tr_Budget = 4*10**9

#Defining the USD scaling factor to avoid numerical issues during optimization
Scalar = 10**4

#Defining negligible costs for transmission line flow and renewable energy
Min_Cost = 0.01/Scalar

#Defining unserved energy penalty
LOL_Penalty = 10000/Scalar

solver = "appsi_highs"

#Defining the number of days in every month
Days_in_months = np.array([31,28,31,30,31,30,31,31,30,31,30,31])

#Defining the number of hours in every month
Hours_in_months = Days_in_months*24

def read_config_file(config_file: str) -> dict:
    with open(config_file, 'r') as f:
        return yaml.safe_load(f)

def prep_parameters(config_file):

    config = read_config_file(config_file)

    #Reading all necessary datasets
    Generators_df = pd.read_csv(config['gen_data_output_file'], header=0)
    Fuel_Prices_df = pd.read_csv(config['fuel_price_output_file'], header=0)
    Gen_Node_Matrix_df = pd.read_csv(config['gen_mat_output_file'], header=0)
    Hydro_max_df = pd.read_csv(config['hydro_max_output_file'], header=0)
    Hydro_min_df = pd.read_csv(config['hydro_min_output_file'], header=0)
    Mustrun_df = pd.read_csv(config['must_run_output_file'], header=0)
    Demand_df = pd.read_csv(config['load_output_file'], header=0)
    Solar_df = pd.read_csv(config['solar_output_file'], header=0)
    Wind_df = pd.read_csv(config['onshore_wind_output_file'], header=0)
    Offshore_Wind_df = pd.read_csv(config['offshore_wind_output_file'], header=0)
    Line_Params_df = pd.read_csv(config['existing_line_param_output_file'], header=0)
    Line_Node_Matrix_df = pd.read_csv(config['line_to_bus_output_file'], header=0)

    Line_Node_Matrix_df = Line_Node_Matrix_df.iloc[:,1:]

    #Defining renewable and thermal generators
    Renewable_Generators = Generators_df.loc[Generators_df['typ'].isin(['solar', 'wind','offshorewind','hydro'])].copy()
    Thermal_Generators = Generators_df.loc[~Generators_df['typ'].isin(['solar', 'wind','offshorewind','hydro'])].copy()

    #Reformatting generation-node matrix to exclude renewable generators
    Gen_Node_Matrix_df = Gen_Node_Matrix_df.loc[Gen_Node_Matrix_df['name'].isin(Thermal_Generators['name'])]
    Gen_Node_Matrix_df = Gen_Node_Matrix_df.iloc[:,1:]

    #Designating number of generators, nodes, lines and demand periods (months of the year = 12)
    Num_Thermal_Gens = Thermal_Generators.shape[0]
    Num_Nodes = Demand_df.shape[1]
    Num_Lines = Line_Params_df.shape[0]
    Num_Periods = len(Hours_in_months)

    #Determining indices
    Thermal_Gens = np.array([*range(0,Num_Thermal_Gens)]) 
    Nodes = np.array([*range(0,Num_Nodes)])
    Lines = np.array([*range(0,Num_Lines)])
    Periods = np.array([*range(0,Num_Periods)])

    #Saving important information as arrays for easy/fast referencing afterwards
    Thermal_Gen_Names = Thermal_Generators["name"].values
    Thermal_Gen_Types = Thermal_Generators["typ"].values
    Thermal_Gen_Node = Thermal_Generators["node"].values
    Thermal_Gen_MaxCap = Thermal_Generators["maxcap"].values
    Thermal_Gen_HeatRate = Thermal_Generators["heat_rate"].values
    Thermal_Gen_VarOM = Thermal_Generators["var_om"].values/Scalar

    Thermal_Fuel_Prices = Fuel_Prices_df.values/Scalar
    Gen_Node_Matrix = Gen_Node_Matrix_df.values

    Node_Names = Demand_df.columns.values
    Hydro_Max = Hydro_max_df.values
    Hydro_Min = Hydro_min_df.values
    Mustrun = Mustrun_df.values
    Demand = Demand_df.values
    Solar = Solar_df.values
    Wind = Wind_df.values
    Offshore_Wind = Offshore_Wind_df.values
    Line_Node_Matrix = Line_Node_Matrix_df.values

    Line_Names = Line_Params_df["line"].values
    Line_Reactances = Line_Params_df["reactance"].values
    Line_Initial_Limits = Line_Params_df["limit"].values
    Line_Types = Line_Params_df["transmission_type"].values
    Line_Lengths = Line_Params_df["length_mile"].values
    Line_Costs = Line_Params_df["inv_cost_$_per_MWmile"].values/Scalar

    return Thermal_Gens, Nodes, Lines, Periods, Thermal_Gen_Names, Thermal_Gen_Types, Thermal_Gen_Node,Thermal_Gen_MaxCap, Thermal_Gen_HeatRate, Thermal_Gen_VarOM, Thermal_Fuel_Prices, Gen_Node_Matrix, Node_Names,Hydro_Max, Hydro_Min,  Solar, Mustrun, Demand, Wind, Offshore_Wind, Line_Node_Matrix, Line_Names, Line_Reactances, Line_Initial_Limits, Line_Types, Line_Lengths, Line_Costs, Num_Thermal_Gens, Num_Nodes, Num_Lines, Num_Periods

    ############################ TRANSMISSION EXPANSION MODEL ############################


def prep_empty_arrays(Num_Periods, Num_Thermal_Gens, Num_Lines, Num_Nodes):

    #Creating empty arrays to store model results
    ThermalGeneration_Results = np.zeros((Num_Periods, Num_Thermal_Gens))
    SolarGeneration_Results = np.zeros((Num_Periods, Num_Nodes))
    WindGeneration_Results = np.zeros((Num_Periods, Num_Nodes))
    OffWindGeneration_Results = np.zeros((Num_Periods, Num_Nodes))
    HydroGeneration_Results = np.zeros((Num_Periods, Num_Nodes))
    PowerFlow_Results = np.zeros((Num_Periods, Num_Lines))
    UnservedEnergy_Results = np.zeros((Num_Periods, Num_Nodes))
    VoltageAngle_Results = np.zeros((Num_Periods, Num_Nodes))
    LineLimit_Results_np = np.zeros(Num_Lines)

    return ThermalGeneration_Results, SolarGeneration_Results, WindGeneration_Results, OffWindGeneration_Results, HydroGeneration_Results, PowerFlow_Results, UnservedEnergy_Results, VoltageAngle_Results, LineLimit_Results_np


def prep_model(Thermal_Gens, Nodes, Lines, Periods):

    #Initializing the model as concrete model
    m=pyo.ConcreteModel()

    #Telling model that we will need duals to check shadow prices for relevant constraints
    m.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)

    #Defining sets
    m.G=pyo.Set(initialize=Thermal_Gens) #All thermal generators
    m.N=pyo.Set(initialize=Nodes) #All nodes
    m.L=pyo.Set(initialize=Lines) #All transmission lines
    m.T=pyo.Set(initialize=Periods) #All demand periods

    #Defining decision variables
    m.ThermalGen = pyo.Var(m.G, m.T, within=pyo.NonNegativeReals, initialize=0) #Generation from thermal generators in MWh
    m.SolarGen = pyo.Var(m.N, m.T, within=pyo.NonNegativeReals, initialize=0) #Generation from solar generators in MWh
    m.WindGen = pyo.Var(m.N, m.T, within=pyo.NonNegativeReals, initialize=0) #Generation from wind generators in MWh
    m.OffWindGen = pyo.Var(m.N, m.T, within=pyo.NonNegativeReals, initialize=0) #Generation from offshore wind generators in MWh
    m.HydroGen = pyo.Var(m.N, m.T, within=pyo.NonNegativeReals, initialize=0) #Generation from hydro generators in MWh

    m.ActualFlow = pyo.Var(m.L, m.T, within=pyo.Reals, initialize=0) #Actual power flow on transmission lines in MWh
    m.DummyFlow = pyo.Var(m.L, m.T, within=pyo.Reals, initialize=0) #Absolute value of power flow on transmission lines in MWh
    m.NewLineCap = pyo.Var(m.L, within=pyo.PositiveReals) #New capacity of each transmission line in MW

    m.LossOfLoad = pyo.Var(m.N, m.T, within=pyo.NonNegativeReals, initialize=0) #Unserved energy in MWh
    m.VoltAngle = pyo.Var(m.N, m.T, within=pyo.Reals, bounds=(-3.1415, 3.1415)) #Voltage angle in radians

    return m

#Defining objective function
def TEPCost(m):

    #Generation cost from dispatchable thermal generators
    Gen_cost = sum(m.ThermalGen[g,t]*((Thermal_Gen_HeatRate[g]*Thermal_Fuel_Prices[t,g])+Thermal_Gen_VarOM[g])*Hours_in_months[t] for g in m.G for t in m.T)
    #Loss of load (i.e., unserved energy cost)
    LOL_cost = sum(m.LossOfLoad[n,t]*LOL_Penalty*Hours_in_months[t] for n in m.N for t in m.T)
    #Generation cost from solar generators
    Solar_cost = sum(m.SolarGen[n,t]*Min_Cost*Hours_in_months[t] for n in m.N for t in m.T)
    #Generation cost from wind generators
    Wind_cost = sum(m.WindGen[n,t]*Min_Cost*Hours_in_months[t] for n in m.N for t in m.T)
    #Generation cost from offshore wind generators
    OffWind_cost = sum(m.OffWindGen[n,t]*Min_Cost*Hours_in_months[t] for n in m.N for t in m.T)
    #Generation cost from hydro generators
    Hydro_cost = sum(m.HydroGen[n,t]*Min_Cost*Hours_in_months[t] for n in m.N for t in m.T)
    #Power flow cost on transmission lines
    Power_flow_cost = sum(m.DummyFlow[l,t]*Min_Cost*Hours_in_months[t] for l in m.L for t in m.T)
    #Investment cost for transmission capacity additions
    Tr_investment_cost = sum((m.NewLineCap[l]-Line_Initial_Limits[l])*Line_Lengths[l]*Line_Costs[l]/60 for l in m.L)

    return Gen_cost + LOL_cost + Solar_cost + Wind_cost + OffWind_cost + Hydro_cost + Power_flow_cost + Tr_investment_cost



def MaxC(m,g,t):
    """Maximum capacity constraint for thermal generators"""
    return m.ThermalGen[g,t] <= Thermal_Gen_MaxCap[g]


def SolarMax(m,n,t):
    """Maximum capacity constraint for solar generators"""
    return m.SolarGen[n,t] <= Solar[t,n]


def WindMax(m,n,t):
    """Maximum capacity constraint for wind generators"""
    return m.WindGen[n,t] <= Wind[t,n]


def OffWindMax(m,n,t):
    """Maximum capacity constraint for offshore wind generators"""
    return m.OffWindGen[n,t] <= Offshore_Wind[t,n]


def HydroMax(m,n,t):
    """Maximum capacity constraint for hydro generators"""
    return m.HydroGen[n,t] <= Hydro_Max[t,n]


def HydroMin(m,n,t):
    """Minimum capacity constraint for hydro generators"""
    return m.HydroGen[n,t] >= Hydro_Min[t,n]


def ThetaRef(m,t):
    """#Voltage angle constraint for reference node"""
    return m.VoltAngle[0,t] == 0


def KVL_Loop(m,l,t):
    """#Kirchhoff's Voltage Law constraint"""
    theta_diff = sum(m.VoltAngle[n,t]*Line_Node_Matrix[l,n] for n in m.N)
    return  100*theta_diff == m.ActualFlow[l,t]*((0.524/Line_Initial_Limits[l])*100)


def FlowUP(m,l,t):
    """#Maximum transmission line flow constraint (positive side)"""
    return m.ActualFlow[l,t] <= m.NewLineCap[l]


def FlowDOWN(m,l,t):
    """#Maximum transmission line flow constraint (negative side)"""
    return -m.ActualFlow[l,t] <= m.NewLineCap[l]


def NewLCapacity(m,l):
    """#Minimum new line capacity constraint"""
    return Line_Initial_Limits[l] <= m.NewLineCap[l]


def DummyFlowUP(m,l,t):
    """#Constraint to find absolute value of flow (positive side)"""
    return  m.DummyFlow[l,t] >= m.ActualFlow[l,t]


def DummyFlowDOWN(m,l,t):
    """#Constraint to find absolute value of flow (negative side)"""
    return  m.DummyFlow[l,t] >= -m.ActualFlow[l,t]

def KCL(m,n,t):
    """Kirchhoff's Current Law (i.e., nodal power balance) constraint"""
    total_power_flow = sum(m.ActualFlow[l,t]*Line_Node_Matrix[l,n] for l in m.L)
    total_thermal_gen = sum(m.ThermalGen[g,t]*Gen_Node_Matrix[g,n] for g in m.G) 
    total_renewable_gen = m.SolarGen[n,t] + m.WindGen[n,t] + m.OffWindGen[n,t] + m.HydroGen[n,t]
    mustrun_gen = Mustrun[t,n]
    total_LOL = m.LossOfLoad[n,t]
    return total_thermal_gen + total_renewable_gen + total_LOL + mustrun_gen - total_power_flow == Demand[t,n]

def TransmissionBudget(m):
    """Constraint to limit yearly transmission investment cost in USD"""
    Tr_inv_cost = sum((m.NewLineCap[l]-Line_Initial_Limits[l])*Line_Lengths[l]*Line_Costs[l] for l in m.L)
    return Tr_inv_cost <= Tr_Budget/Scalar


#Starting TEP simulation
config_file = my_config_file_path

# read the config file
config_dict = read_config_file(config_file)

# set up output file paths
thermal_generation_output_path = config_dict['tep_output_thermal_gen_file']
solar_generation_output_path = config_dict['tep_output_solar_gen_file']
wind_generation_output_path = config_dict['tep_output_wind_gen_file']
offshore_wind_generation_output_path = config_dict['tep_output_offshore_wind_gen_file']
hydro_generation_output_path = config_dict['tep_output_hydro_gen_file']
unserved_energy_output_path = config_dict['tep_output_unserved_energy_file']
voltage_angle_output_path = config_dict['tep_output_volt_angle_file']
power_flow_output_path = config_dict['tep_output_power_flow_file']
line_limits_output_path = config_dict['tep_output_new_transmission_file']
tep_line_file_output_path = config_dict['tep_output_line_params_file']
go_run_input_path = config_dict['go_input_line_params_file']

apply_transmission_budget = bool(config_dict['apply_budget_bool'])

Thermal_Gens, Nodes, Lines, Periods, Thermal_Gen_Names, Thermal_Gen_Types, Thermal_Gen_Node,Thermal_Gen_MaxCap, Thermal_Gen_HeatRate, Thermal_Gen_VarOM, Thermal_Fuel_Prices, Gen_Node_Matrix, Node_Names, Hydro_Max, Hydro_Min,  Solar, Mustrun, Demand, Wind, Offshore_Wind, Line_Node_Matrix, Line_Names, Line_Reactances, Line_Initial_Limits, Line_Types, Line_Lengths, Line_Costs,  Num_Thermal_Gens, Num_Nodes, Num_Lines, Num_Periods = prep_parameters(config_file=config_file)

ThermalGeneration_Results, SolarGeneration_Results, WindGeneration_Results, OffWindGeneration_Results, HydroGeneration_Results, PowerFlow_Results, UnservedEnergy_Results, VoltageAngle_Results, LineLimit_Results_np = prep_empty_arrays(Num_Periods=Num_Periods, Num_Thermal_Gens=Num_Thermal_Gens, Num_Lines=Num_Lines, Num_Nodes=Num_Nodes)

#Defining solver
opt = pyo.SolverFactory(solver)

m = prep_model(Thermal_Gens, Nodes, Lines, Periods)

m.ObjectiveFunc=pyo.Objective(rule=TEPCost, sense=pyo.minimize)

#Defining constraints
m.ThermalMaxCap_Cons= pyo.Constraint(m.G, m.T, rule=MaxC)
m.SolarMaxCap_Cons= pyo.Constraint(m.N, m.T, rule=SolarMax)
m.WindMaxCap_Cons= pyo.Constraint(m.N, m.T, rule=WindMax)
m.OffshoreWindMaxCap_Cons= pyo.Constraint(m.N, m.T, rule=OffWindMax)
m.HydroMaxCap_Cons= pyo.Constraint(m.N, m.T, rule=HydroMax)
m.HydroMinCap_Cons= pyo.Constraint(m.N, m.T, rule=HydroMin)
m.RefVoltAngle_Cons = pyo.Constraint(m.T, rule=ThetaRef)
m.KVLAroundLoopConstraint = pyo.Constraint(m.L, m.T, rule=KVL_Loop)
m.FlowUP_Cons = pyo.Constraint(m.L, m.T ,rule=FlowUP)
m.FlowDOWN_Cons = pyo.Constraint(m.L, m.T ,rule=FlowDOWN)
m.NewLCapacity_Cons = pyo.Constraint(m.L, rule=NewLCapacity)
m.DummyFlowUP_Cons = pyo.Constraint(m.L, m.T ,rule=DummyFlowUP)
m.DummyFlowDOWN_Cons = pyo.Constraint(m.L, m.T ,rule=DummyFlowDOWN)

#Kirchhoff's Current Law (i.e., nodal power balance) constraint
m.KCL_Cons = pyo.Constraint(m.N, m.T, rule=KCL)

if apply_transmission_budget:
    #Constraint to limit yearly transmission investment cost in USD
    m.TransmissionBudget_Cons = pyo.Constraint(rule=TransmissionBudget)
else:
    pass

#Calling the solver to solve the model
TEP_results = opt.solve(m, tee=True)

#Checking the solver status and if solution is feasible or not
if (TEP_results.solver.status == pyo.SolverStatus.ok) and (TEP_results.solver.termination_condition == pyo.TerminationCondition.optimal):
    print("\033[92mSuccess! Solution is feasible. \033[00m")
elif (TEP_results.solver.termination_condition == pyo.TerminationCondition.infeasible):
    print("\033[91mSolution is INFEASIBLE!!! \033[00m")
else:
    print(f"\033[91mSomething else is not right, solver status is {TEP_results.solver.status}. \033[00m")

#Saving and writing thermal generation results
for g_n in Thermal_Gens:
    for t_n in Periods:
        ThermalGeneration_Results[t_n, g_n] = m.ThermalGen[g_n, t_n]()

ThermalGeneration_Results = pd.DataFrame(ThermalGeneration_Results, columns=Thermal_Gen_Names)
ThermalGeneration_Results.to_csv(thermal_generation_output_path, index=False)

#Saving and writing solar generation results
for n_n in Nodes:
    for t_n in Periods:
        SolarGeneration_Results[t_n, n_n] = m.SolarGen[n_n, t_n]()

SolarGeneration_Results = pd.DataFrame(SolarGeneration_Results, columns=Node_Names)
SolarGeneration_Results.to_csv(solar_generation_output_path, index=False)

#Saving and writing wind generation results
for n_n in Nodes:
    for t_n in Periods:
        WindGeneration_Results[t_n, n_n] = m.WindGen[n_n, t_n]()

WindGeneration_Results = pd.DataFrame(WindGeneration_Results, columns=Node_Names)
WindGeneration_Results.to_csv(wind_generation_output_path, index=False)

#Saving and writing offshore wind generation results
for n_n in Nodes:
    for t_n in Periods:
        OffWindGeneration_Results[t_n, n_n] = m.OffWindGen[n_n, t_n]()

OffWindGeneration_Results = pd.DataFrame(OffWindGeneration_Results, columns=Node_Names)
OffWindGeneration_Results.to_csv(offshore_wind_generation_output_path, index=False)

#Saving and writing hydro generation results
for n_n in Nodes:
    for t_n in Periods:
        HydroGeneration_Results[t_n, n_n] = m.HydroGen[n_n, t_n]()

HydroGeneration_Results = pd.DataFrame(HydroGeneration_Results, columns=Node_Names)
HydroGeneration_Results.to_csv(hydro_generation_output_path, index=False)

#Saving and writing unserved energy results
for n_n in Nodes:
    for t_n in Periods:
        UnservedEnergy_Results[t_n, n_n] = m.LossOfLoad[n_n, t_n]()

UnservedEnergy_Results = pd.DataFrame(UnservedEnergy_Results, columns=Node_Names)
UnservedEnergy_Results.to_csv(unserved_energy_output_path, index=False)

#Saving and writing voltage angle results
for n_n in Nodes:
    for t_n in Periods:
        VoltageAngle_Results[t_n, n_n] = m.VoltAngle[n_n, t_n]()

VoltageAngle_Results = pd.DataFrame(VoltageAngle_Results, columns=Node_Names)
VoltageAngle_Results.to_csv(voltage_angle_output_path, index=False)

#Saving and writing power flow results
for l_n in Lines:
    for t_n in Periods:
        PowerFlow_Results[t_n, l_n] = m.ActualFlow[l_n, t_n]()

PowerFlow_Results = pd.DataFrame(PowerFlow_Results, columns=Line_Names)
PowerFlow_Results.to_csv(power_flow_output_path, index=False)

#Saving and writing new transmission line limit results
for l_n in Lines:
    LineLimit_Results_np[l_n] = m.NewLineCap[l_n]()

LineLimit_Results = pd.DataFrame(LineLimit_Results_np, columns=["New_Capacity"])
LineLimit_Results.insert(0, "Name", Line_Names)
LineLimit_Results.insert(1, "Reactance", (0.524/LineLimit_Results_np)*100)
LineLimit_Results.insert(2, "Type", Line_Types)
LineLimit_Results.insert(3, "Length", Line_Lengths)
LineLimit_Results.insert(4, "Capital_Cost", Line_Costs*Scalar)
LineLimit_Results.insert(5, "Old_Capacity", Line_Initial_Limits)
LineLimit_Results["Capacity_Addition"] = LineLimit_Results["New_Capacity"] - LineLimit_Results["Old_Capacity"]
LineLimit_Results["Total_Investment"] = LineLimit_Results["Capital_Cost"]*LineLimit_Results["Capacity_Addition"]*LineLimit_Results["Length"]
LineLimit_Results.to_csv(line_limits_output_path, index=False)

GO_Line_File = pd.DataFrame(LineLimit_Results_np, columns=["limit"])
GO_Line_File.insert(0, "line", Line_Names)
GO_Line_File.insert(1, "reactance", (0.524/LineLimit_Results_np)*100)

# save to output folder and run folder
GO_Line_File.to_csv(tep_line_file_output_path, index=False)
GO_Line_File.to_csv(go_run_input_path, index=False)