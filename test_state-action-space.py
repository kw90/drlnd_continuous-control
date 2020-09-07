from unityagents import UnityEnvironment
import numpy as np

environment = UnityEnvironment(file_name='./Reacher_Linux/Reacher.x86_64')
# Get the default brain
brain_name = environment.brain_names[0]
brain = environment.brains[brain_name]

# Reset the environment
environment_info = environment.reset(train_mode=True)[brain_name]

# Number of agents
num_agents = len(environment_info.agents)
print('Number of agents:', num_agents)

# Size of each action
action_size = brain.vector_action_space_size
print('Size of each action:', action_size)

# Examine the state space 
states = environment_info.vector_observations
state_size = states.shape[1]
print('There are {} agents. Each observes a state with length: {}'.format(states.shape[0], state_size))
print('The state for the first agent looks like:', states[0])

environment.close()
