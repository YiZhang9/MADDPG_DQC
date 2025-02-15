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
    "MADDPG_Agent": {
        "learning_rate": 0.0001,
        "batch_size": 1024,
        "buffer_size": 1000000,
        "gamma": 0.99,  # Discount factor
        "tau": 0.01,  # Soft update parameter
        "update_every_n_steps": 5,
        "num_units": 64,  # Number of hidden units
        "actor_lr": 0.001,
        "critic_lr": 0.001,
        "max_episode_length": 25,
        "exploration": 0.1,  # Exploration factor
        "clip_rewards": False
    }
}

# Initialize Replay Buffer
replay_buffer = ReplayBuffer(size=config.hyperparameters["MADDPG_Agent"]["buffer_size"])


if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')

    # Define the agent using MADDPG
    agent = MADDPGAgentTrainer(
        name="MADDPG_Agent",
        model=None,  # Replace with your model function
        obs_shape_n=[config.environment.state_size] * config.environment.num_agents,
        act_space_n=[config.environment.action_size] * config.environment.num_agents,
        agent_index=0,
        args=config.hyperparameters["MADDPG_Agent"],
        local_q_func=False
    )

    # Sample training loop (Modify based on your requirements)
    for episode in range(config.num_episodes_to_run):
        obs = config.environment.reset()
        done = False
        episode_reward = 0

        while not done:
            action = agent.action(obs)  # Get action from MADDPG agent
            new_obs, reward, done, _ = config.environment.step(action)

            # Store experience in replay buffer
            replay_buffer.add(obs, action, reward, new_obs, done)

            # Train the agent if enough samples are collected
            if len(replay_buffer) > config.hyperparameters["MADDPG_Agent"]["batch_size"]:
                agent.update([agent], episode)

            obs = new_obs
            episode_reward += reward

        print(f"Episode {episode + 1}, Reward: {episode_reward}")

    print("Training complete!")



    
    
