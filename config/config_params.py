#
# A Configuration file for the Shared Paramters.
#

# Imports from Standard Library 
import os
from collections import defaultdict

# Imports from Third Party Libraries
# from FA_model import NeuralNet


CONFIG = {
    # setup ID for this configuration
    'setup': 'Parameter_setup_1',
    
    # Is it tabular or FA
    'tabular': True,
    'function_approximation': False,
    
    # Parameters for the Agent 
    "eps_per_evaluation": 1, # change to eps per GPI iter
    "num_GPI_iter": 5,
    "GPI_save_freq": 5,
    "test_episodes": 5,
    "gamma": 0.3,
    "epsilon_start": 1.0,
    "epsilon_min": 0.05,
    "epsilon_decay": 0.9,

    
    # Parameters for the BSE
    "bse_sys_minprice" : 1,
    "bse_sys_maxprice" : 5,
    
}
    
#
# Dictionary is initialised now can self reference if needed. 
#

#### Parameters for the Agent
CONFIG["total_eps"] = CONFIG["eps_per_evaluation"] * CONFIG["num_GPI_iter"] # total number of episodes
CONFIG["GPI_test_freq"] = CONFIG["GPI_save_freq"] * 1 # how often we test, must be a multiple of GPI_save_freq since we need to load CSV file for testing.
CONFIG["action_space"] =  [n/1.0 for n in range(CONFIG["bse_sys_maxprice"] - CONFIG["bse_sys_minprice"] + 1)]  # This allows any price on the BSE. Determined by the BSE min and max.


############### Parameters Depending on the type of RL agent ########################

## tabular Specific Parameters : ##
if CONFIG['tabular']: 
    CONFIG_tab = {
    
    "initial q_table": defaultdict(lambda: 0.0),
    "alpha" : 0.1,
    "no._of_bins": 5,
    
    
    # Observation space 
    "order" : True, # this will always be true
    "best": True,
    "worst": False,
    "average": False,
    "std": False,
    "total_orders": False,
    "time_left": False, 
    "binary_flag" : False,
       
    
    ### file paths
    'q_tables' :  os.path.join('tab_' + CONFIG["setup"], 'q_tables'),
    'counts' :  os.path.join('tab_' + CONFIG["setup"], 'counts'),
    'plots' :  os.path.join('tab_' +  CONFIG["setup"], 'plots'),
    # Full path for the Parameters.py file (inside the setup_1 folder)
    'parameters': os.path.join('tab_' + CONFIG["setup"], 'Parameters.py'),
    }
    
    CONFIG = CONFIG | CONFIG_tab # combine to the main dictionary
    

    sellers_spec = [('RL_tabular', 1, {'epsilon': CONFIG['epsilon_start'], 'action_space': CONFIG['action_space'], 'q_table_seller': CONFIG['initial q_table']}),('GVWY',19) ]
    # Loop through to find the index of the RL agent
    for i, seller in enumerate(sellers_spec):
        if seller[0] == 'RL_tabular':  # Check if the first element matches 'RL_tabular'
            CONFIG['rl_index'] = i
            break  # Exit the loop once found
        
        
        
        
        
        
## Function Approximation Specific Parameters : ##
elif CONFIG['function_approximation']:
    CONFIG_FA = {
    
    
    # "n_neurons_hl1" : 120,
    # "n_neurons_hl2" : 64,
    
    # Observation space 
    "order" : True, # this will always be true
    "best": False,
    "worst": False,
    "average": False,
    "std": False,
    "total_orders": True,
    "time_left": True, 
    "binary_flag" : True,
    
    
    ### file paths
    'weights':  os.path.join('FA_' + CONFIG["setup"], 'weights_and_biases'),
    'plots' :  os.path.join('FA_' +  CONFIG["setup"], 'plots'),
     # Full path for the Parameters.py file (inside the setup_1 folder)
    'parameters' : os.path.join('FA_' + CONFIG["setup"], 'Parameters.py'),
   }
    
    # calculate the dimension of the (state,action) space.
    keys_to_check = ["order", "best", "worst", "average", "std", "total_orders", "time_left", "binary_flag"]
    n_obs_features = sum(CONFIG[key] for key in keys_to_check if key in CONFIG)
    CONFIG["n_features"] = n_obs_features + len(CONFIG['action_space']) # this is the length of an observation element (plus number of actions for one hot encoding) 

    CONFIG["nn_dims"] = (CONFIG["n_features"], 64, 32, 1)

    CONFIG = CONFIG | CONFIG_FA # combine to the main dictionary
    
    sellers_spec = [('GVWY',19), ('RL_FA', 1, {'epsilon': CONFIG['epsilon_start'], 'action_space': CONFIG['action_space'],'norm_params': {
        "x_min": 0,
        "x_max": 1,
        "y_min": 0,
        "y_max": 1,
    }})]
    # Loop through to find the index of the RL agent
    for i, seller in enumerate(sellers_spec):
        if seller[0] == 'RL_FA':  # Check if the first element matches 'RL_FA'
            CONFIG['rl_index'] = i
            break  # Exit the loop once found
    
    
    

    
    
    
    
    
    
    
########################################## Market Parameters ####################################################
sess_id = 'session_1'
start_time = 0.0
end_time = 30.0


buyers_spec = [('GVWY',20)]

trader_spec = {'sellers': sellers_spec, 'buyers': buyers_spec}

range1 = (1, 5)
supply_schedule = [{'from': start_time, 'to': end_time, 'ranges': [range1], 'stepmode': 'fixed'}]

range2 = (1, 5)
demand_schedule = [{'from': start_time, 'to': end_time, 'ranges': [range2], 'stepmode': 'fixed'}]

# new customer orders arrive at each trader approx once every order_interval seconds
order_interval = 30

order_schedule = {'sup': supply_schedule, 'dem': demand_schedule,
                'interval': order_interval, 'timemode': 'drip-fixed'}

dump_flags = {'dump_strats': True, 'dump_lobs': True, 'dump_avgbals': True, 'dump_tape': True, 'dump_blotters': True}
verbose = False

# save these to our config dictionary
CONFIG["market_params"] = (sess_id, start_time, end_time, trader_spec, order_schedule, dump_flags, verbose)
