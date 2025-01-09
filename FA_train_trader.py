import random
import csv
import numpy as np 


from tqdm import tqdm
from BSE import market_session
from collections import defaultdict
from typing import List, Dict, DefaultDict, Tuple
from epsilon_scheduling import linear_epsilon_decay
from tab_converting_csv_and_dictionary import save_q_table_dict_to_csv, save_sa_counts_to_csv


import ast

# File handling
import shutil
import csv
import os

# Importing Global Parameters
from config.config_params import CONFIG

# Neural Network Imports
from FA_model import NeuralNet, train_network, normalize_data_min_max
from torch import optim
from torch import nn
import torch

# Import the logging module and create a module-level logger
import logging
logger = logging.getLogger(__name__)

###### The functions


def To_data_gradient_MC_with_returns(
    obs_list: List[List[int]],
    action_list: List[int],
    reward_list: List[float]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Transform the observed trajectories into data for the gradient Monte Carlo algorithm,
    converting the data directly into PyTorch tensors. The target data (y_tensor) contains
    the returns (discounted rewards).

    Parameters:
        obs_list (List[List[int]]): List of observations (states), where each observation is a list of integers.
        action_list (List[int]): List of actions taken.
        reward_list (List[float]): List of rewards received.

    Returns:
        x_tensor (torch.Tensor): Input data tensor of shape (batch_size, input_size).
        y_tensor (torch.Tensor): Target data tensor of shape (batch_size,).
    """
    # Step 1: Compute returns (discounted rewards)
    traj_length = len(reward_list)
    G = [0 for _ in range(traj_length)]
    
    if traj_length > 0:
        G[-1] = reward_list[-1]
        for t in range(traj_length - 2, -1, -1):
            G[t] = reward_list[t] + CONFIG["gamma"] * G[t + 1]

    # Step 2: Combine observation and action for each step
    processed = []
    for obs, action in zip(obs_list, action_list):
        processed.append(obs + [action])  # Combine observation and action

    # Step 3: Convert to PyTorch tensors
    x_tensor = torch.tensor(processed, dtype=torch.float)
    y_tensor = torch.tensor(G, dtype=torch.float)  # Use the computed returns as targets
    y_tensor = y_tensor.unsqueeze(1)  # Converts shape to 2D

    
    return x_tensor, y_tensor


def train(total_eps: int, market_params: tuple, epsilon_start: float) :
    """
    Train the function approximationn RL agent over a specified number of episodes.

    Parameters:
        total_eps (int): Total number of episodes to train.
        market_params (tuple): Parameters for the market session.
        epsilon_start (float): Starting value of epsilon for exploration.

    Returns:
        Doesnt return anything just saves CSV file containing parameters of network at
        different GPI iterations for later analysis. 
    """
    ## Initialize everything: ##
    
    # Start the GPI iterations at 1 
    GPI_iter = 1
    logger.info(f"Starting GPI iteration {GPI_iter}")
    
    # initialize the epsilon
    epsilon = epsilon_start
    
    # initialise the data as tensors for pytorch.
    inputs = torch.empty((0, CONFIG["n_features"]), dtype=torch.float32)
    targets = torch.empty((0, 1), dtype=torch.float32)
    normparams = {"x_min": 0, "x_max": 1, "y_min": 0, "y_max": 1}
    
    # initialise the model
    neural_network = NeuralNet(dims=CONFIG["nn_dims"])
    optimizer = optim.Adam(neural_network.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    # initialise the neural network
    market_params[3]['sellers'][CONFIG['rl_index']][2]['neural_net'] = neural_network
    
    ## Run the GPI iterations: ##
    
    for episode in range(1, total_eps + 1):
        # Run the market session once, it returns a list of observations, actions and rewards for the RL trader 
        obs_list, action_list, reward_list = market_session(*market_params)
       
        # Calculate returns and Transform to tensors for the neural network 
        
        try:
            more_inputs, more_targets = To_data_gradient_MC_with_returns(obs_list, action_list, reward_list)    
            # Add to the current data under this policy
            print(f"more_inputs shape: {more_inputs.shape}")
            print(f"more_targets shape: {more_targets.shape}")
            print(f"inputs shape before concat: {inputs.shape}")
            print(f"targets shape before concat: {targets.shape}")

            inputs = torch.cat((inputs, more_inputs), 0)
            targets = torch.cat((targets, more_targets), 0)
            
            
        except Exception as e:
            logger.error(f"Error using data to calculate returns and transform to tensors in episode {episode}: {e}")
            

        # If we have done the designated number of episodes for this policy evaluation,
        # retrain the network and get the new parameters.
        if episode % CONFIG["eps_per_evaluation"] == 0: 
            # Normalize the data before training 
            try:
                inputs, targets, normparams = normalize_data_min_max(inputs, targets) # normalises both the state and the action together as inputs, and then the return as targets
                
            except Exception as e:
                logger.error(f"Error normalizing data in GPI iter {GPI_iter}: {e}")
                
            # Retrain the network 
            try:
                train_network(neural_network, optimizer, criterion, inputs, targets)
            except Exception as e:
                logger.error(f"Error training in GPI iter {GPI_iter}: {e}")
            
            
            
            # update the market parameter with the newest neural network
            market_params[3]['sellers'][CONFIG['rl_index']][2]['neural_net'] = neural_network
            # pass in norm params
            market_params[3]['sellers'][CONFIG['rl_index']][2]['norm_params'] = normparams



            #### Save Network (for later inspection)  ####
            if episode % CONFIG["GPI_save_freq"] == 0:
                logger.info(f"Saving the Neural Network for GPI iter {GPI_iter}")
                file_path = os.path.join(CONFIG["weights"], f'network_at_GPI_{GPI_iter}.pth')
                try:
                    torch.save(neural_network.state_dict(), file_path)
                    logger.info(f"Neural network parameters saved to {file_path}")
                except Exception as e:
                    logger.error(f"Error saving neural network parameters in GPI iter {GPI_iter}, episode {episode}: {e}")
            #### End of saving ####
            
            # Update epsilon for the next iteration of policy evaluation
            epsilon = linear_epsilon_decay(
                GPI_iter, 
                CONFIG["num_GPI_iter"], 
                epsilon_start, 
                CONFIG["epsilon_min"], 
                CONFIG["epsilon_decay"])
            market_params[3]['sellers'][CONFIG['rl_index']][2]['epsilon'] = epsilon
            logger.info(f"New epsilon: {epsilon}")
            

            # reset the data
            inputs = torch.empty((0, CONFIG["n_features"]), dtype=torch.float32)
            targets = torch.empty((0, 1), dtype=torch.float32)
            
            GPI_iter += 1
            logger.info(f"Starting GPI iteration {GPI_iter}")
    
    return  
