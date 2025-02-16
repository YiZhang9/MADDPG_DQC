import sys
import os
from os.path import dirname, abspath


# Add the parent directory of the project to Python's module search path
sys.path.append(dirname(dirname(abspath(__file__))))  # Adds the grandparent directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))  # Adds the parent directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "environments")))  # Adds the 'environments' folder
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "MADDPG")))  # Adds MADDPG directory



from MADDPG.MADDPG import MADDPGAgentTrainer
from environments.SystemStateClass import SystemStateClass
from environments.QuantumEnv import EnvUpdater

# Initialize Quantum Environment
env = EnvUpdater(completion_deadline=1500)

# Create an agent
agent = MADDPGAgentTrainer(
    name="MADDPG_Agent",
    model=None,  # Replace with your model
    obs_shape_n=[env.state_size] * 2,
    act_space_n=[env.action_size] * 2,
    agent_index=0,
    args={},
    local_q_func=False
)

print("Agent initialized successfully!")
