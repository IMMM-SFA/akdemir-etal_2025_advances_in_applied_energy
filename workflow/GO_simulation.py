import go
from go import Model
import sys

#Defining number of simulation days and path to the config file
my_config_file_path = "GO_config.yml"
my_simulation_days = 365
my_solver_name = "appsi_highs"


def run_go(config_file_path:str, simulation_days:int, solver_name:str):
    """
    Run the GO model with the given configuration file for the indicated number of days.

    :param: config_file_path                    path to GO configuration file
    :param: simulation_days                     number of simulation days to run for

    """

    # build the data (.dat) file
    go.build_data_file(config_file=config_file_path, simulation_days=simulation_days)


    # instantiate go model
    model = Model(
        region="west",
        problem="linear",
        complexity="multi",
        solver_name=solver_name,
        solver_params={
            "solver": "choose",
            "parallel": "on",
            "threads": 0, # all available
            "random_seed": 10, 
            "time_limit": 3600 * 1000,
            "run_crossover": "on",
        }
    )
    # run the model
    model.run(
        config_file=config_file_path, 
        n_days=simulation_days,
        retry_solver_list=["simplex", "ipm"],
        retry_n_seeds=1,
        retry_dual_feasibility_tolerance_list=[1e-06,1e-04],
        retry_primal_feasibility_tolerance_list=[1e-06, 1e-04],
        retry_ipm_optimality_tolerance_list=[1e-06, 1e-04],
        reset_restart_file=True
    )

    return None



#Starting GO simulation
run_go(config_file_path = my_config_file_path, simulation_days = my_simulation_days, solver_name = my_solver_name)