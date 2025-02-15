import os
import sys
from os.path import dirname, abspath

# Add the parent directory of the project to Python's module search path
sys.path.append(dirname(dirname(abspath(__file__))))  # Adds the grandparent directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))  # Adds the parent directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "environments")))  # Adds the 'environments' folder
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "MADDPG")))  # Adds MADDPG directory


from MADDPG.MADDPG import MADDPGAgentTrainer
from MADDPG.replay_buffer import ReplayBuffer  # Importing ReplayBuffer from its file

# Import EnvUpdater from the QuantumEnv module located in the 'environments' folder
from environments.QuantumEnv import EnvUpdater  

# Import different reinforcement learning (RL) agent classes from various subdirectories in the project

# Actor-Critic RL agents
# from agents.actor_critic_agents.A2C import A2C  # Advantage Actor-Critic (A2C)
# from agents.actor_critic_agents.DDPG import DDPG  # Deep Deterministic Policy Gradient (DDPG)
# from agents.actor_critic_agents.SAC_Discrete import SAC_Discrete  # Soft Actor-Critic (SAC) for discrete action spaces
# from agents.actor_critic_agents.A3C import A3C  # Asynchronous Advantage Actor-Critic (A3C)

# # Deep Q-Network (DQN) based agents
# from agents.DQN_agents.Dueling_DDQN import Dueling_DDQN  # Dueling Double DQN (DDQN)
# from agents.DQN_agents.DDQN import DDQN  # Double DQN
# from agents.DQN_agents.DQN import DQN  # Standard Deep Q-Network (DQN)
# from agents.DQN_agents.DQN_With_Fixed_Q_Targets import DQN_With_Fixed_Q_Targets  # DQN with fixed Q-targets
# from agents.DQN_agents.DDQN_With_Prioritised_Experience_Replay import DDQN_With_Prioritised_Experience_Replay  # DDQN with Prioritized Experience Replay

# # Policy Gradient RL agents
# from agents.policy_gradient_agents.PPO import PPO  # Proximal Policy Optimization (PPO)
# from agents.policy_gradient_agents.REINFORCE import REINFORCE  # REINFORCE policy gradient method

# # Training module
# from agents.Trainer import Trainer  # Generalized trainer module for training RL agents

# Utility functions and configurations
from utilities.data_structures.Config import Config  # Configuration file handler for RL experiments

import torch


config = Config()

config.seed = 123453
config.num_episodes_to_run = 250   # control number of episodes was 60
config.file_to_save_data_results = "results/data_and_graphs/dist_quantum_Results_Data.pkl"   #save results 
config.file_to_save_results_graph = "results/data_and_graphs/dist_quantum__Results_Graph.png"   #save graph
config.show_solution_score = False
config.visualise_individual_results = False
config.visualise_overall_agent_results = True
config.baselines = True
config.standard_deviation_results = 1.0
config.runs_per_agent = 1
config.use_GPU = True
config.overwrite_existing_results_file = False
config.randomise_random_seed = True
config.save_model = True

# line below, 
# state_size is number of physcial qubit locations in processors (directload TBD), 
# completion_deadline is time by which DAG must be completed
config.environment = EnvUpdater(completion_deadline = 1500 - 1)  #1500  # how many steps we allow for the DAG to be executed


config.hyperparameters = {
    "DQN_Agents": {
        "learning_rate": 0.00001,  #working was 0.00001          #was 0.001 have tried 0.0001
        "batch_size": 256*10, #256*10,
        "buffer_size": 100000, #was 80000
        "epsilon": 1.0,
        "epsilon_decay_rate_denominator": 80, #was 50 
        "discount_rate": 0.99,  #0.99,
        "tau": 0.001,
        "update_every_n_steps": 5,
        "linear_hidden_units": [140,150],     #working was [90,80] and before that [70, 80] did not work [250,150]
        "final_layer_activation": "None",
        "batch_norm": False,
        "gradient_clipping_norm": 0.7,
        "learning_iterations": 10,
        "clip_rewards": False
    },
    "Stochastic_Policy_Search_Agents": {
        "policy_network_type": "Linear",
        "noise_scale_start": 1e-2,
        "noise_scale_min": 1e-3,
        "noise_scale_max": 2.0,
        "noise_scale_growth_factor": 2.0,
        "stochastic_action_decision": False,
        "num_policies": 10,
        "episodes_per_policy": 1,
        "num_policies_to_keep": 5,
        "clip_rewards": False
    },
    "Policy_Gradient_Agents": {
        "learning_rate": 0.00005,
        "linear_hidden_units": [70,180,70],
        "final_layer_activation": "SOFTMAX",
        "learning_iterations_per_round": 20,
        "discount_rate": 0.992,
        "batch_norm": False,
        "clip_epsilon": 0.2,
        "episodes_per_learning_round": 30, 
        "normalise_rewards": True,
        "gradient_clipping_norm": 7.0,
        "mu": 0.0, #only required for continuous action games
        "theta": 0.0, #only required for continuous action games
        "sigma": 0.0, #only required for continuous action games
        "epsilon_decay_rate_denominator": 50,
        "clip_rewards": False
    },

    "Actor_Critic_Agents":  {

        "learning_rate": 0.005,
        "linear_hidden_units": [70,80,70],
        "final_layer_activation": ["SOFTMAX", None],
        "gradient_clipping_norm": 5.0,
        "discount_rate": 0.99,
        "epsilon_decay_rate_denominator": 1.0,
        "normalise_rewards": True,
        "exploration_worker_difference": 2.0,
        "clip_rewards": False,

        "Actor": {
            "learning_rate": 0.0003,
            "linear_hidden_units": [64, 64, 64],
            "final_layer_activation": "Softmax",
            "batch_norm": False,
            "tau": 0.005,
            "gradient_clipping_norm": 5,
            "initialiser": "Xavier"
        },

        "Critic": {
            "learning_rate": 0.0003,
            "linear_hidden_units": [64, 64, 64],
            "final_layer_activation": None,
            "batch_norm": False,
            "buffer_size": 1000000,
            "tau": 0.005,
            "gradient_clipping_norm": 5,
            "initialiser": "Xavier"
        },

        "min_steps_before_learning": 400,
        "batch_size": 256,
        "discount_rate": 0.99,
        "mu": 0.0, #for O-H noise
        "theta": 0.15, #for O-H noise
        "sigma": 0.25, #for O-H noise
        "action_noise_std": 0.2,  # for TD3
        "action_noise_clipping_range": 0.5,  # for TD3
        "update_every_n_steps": 1,
        "learning_updates_per_learning_session": 4,
        "automatically_tune_entropy_hyperparameter": True,
        "entropy_term_weight": None,
        "add_extra_noise": False,
        "do_evaluation_iterations": True
    }
}

if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')
    AGENTS = [DDQN]  # the value for values, try [DQN] or [PPO] (I think [DDQN] it's fine) 
    
    #[DQN]  #[DDQN]  #[SAC_Discrete, DDQN, Dueling_DDQN, DQN, DQN_With_Fixed_Q_Targets,SNN_HRL, SAC, DDPG, 
              #DDQN_With_exPrioritised_Experience_Replay, A2C, PPO, A3C ]
    trainer = Trainer(config, AGENTS)
    trainer.run_games_for_agents()





    #     "DQN_Agents": {
    #     "learning_rate": 0.00001,  #working was 0.00001          #was 0.001 have tried 0.0001
    #     "batch_size": 256*10, #256*10,
    #     "buffer_size": 100000, #was 80000
    #     "epsilon": 1.0,
    #     "epsilon_decay_rate_denominator": 50, #was 30 
    #     "discount_rate": 0.99,  #0.99,
    #     "tau": 0.001,


    #     "update_every_n_steps": 5,
    #     "linear_hidden_units": [140,150],     #working was [90,80] and before that [70, 80]
    #     "final_layer_activation": "None",
    #     "batch_norm": False,
    #     "gradient_clipping_norm": 0.7,
    #     "learning_iterations": 10,
    #     "clip_rewards": False
    # },









    #     "DQN_Agents": {
    #     "learning_rate": 0.00001,  #working was 0.00001          #was 0.001 have tried 0.0001
    #     "batch_size": 256*10,
    #     "buffer_size": 100000, #was 80000
    #     "epsilon": 1.0,
    #     "epsilon_decay_rate_denominator": 50, #was 30 
    #     "discount_rate": 0.99,  #0.99,
    #     "tau": 0.001,


    #     "update_every_n_steps": 5,
    #     "linear_hidden_units": [90,80],     #working was [70, 80]
    #     "final_layer_activation": "None",
    #     "batch_norm": False,
    #     "gradient_clipping_norm": 0.7,
    #     "learning_iterations": 10,
    #     "clip_rewards": False
    # },






    #     "DQN_Agents": {
    #     "learning_rate": 0.00001,  #working was 0.00001          #was 0.001 have tried 0.0001
    #     "batch_size": 256*10,
    #     "buffer_size": 100000, #was 80000
    #     "epsilon": 1.0,
    #     "epsilon_decay_rate_denominator": 50, #was 30 
    #     "discount_rate": 0.99,  #0.99,
    #     "tau": 0.001,


    #     "update_every_n_steps": 5,
    #     "linear_hidden_units": [90,80],     #working was [70, 80]
    #     "final_layer_activation": "None",
    #     "batch_norm": False,
    #     "gradient_clipping_norm": 0.7,
    #     "learning_iterations": 10,
    #     "clip_rewards": False
    # },



    # "DQN_Agents": {
    #     "learning_rate": 0.00001,  #working was 0.00001          #was 0.001 have tried 0.0001
    #     "batch_size": 256*10,
    #     "buffer_size": 100000, #was 80000
    #     "epsilon": 1.0,
    #     "epsilon_decay_rate_denominator": 50, #was 30 
    #     "discount_rate": 0.99,  #0.99,
    #     "tau": 0.01,


    #     "update_every_n_steps": 5,
    #     "linear_hidden_units": [90,80],     #working was [70, 80]
    #     "final_layer_activation": "None",
    #     "batch_norm": False,
    #     "gradient_clipping_norm": 0.7,
    #     "learning_iterations": 10,
    #     "clip_rewards": False
    # },
    
    
