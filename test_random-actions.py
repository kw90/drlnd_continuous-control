from unityagents import UnityEnvironment
import numpy as np

environment = UnityEnvironment(file_name='./Reacher_Linux/Reacher.x86_64')
# Get the default brain
brain_name = environment.brain_names[0]
brain = environment.brains[brain_name]

# Reset the environment
environment_info = environment.reset(train_mode=False)[brain_name]

# Number of agents
num_agents = len(environment_info.agents)
print('Number of agents:', num_agents)

# Size of each action
action_size = brain.vector_action_space_size
print('Size of each action:', action_size)

states = environment_info.vector_observations                  # get the current state (for each agent)
scores = np.zeros(num_agents)                          # initialize the score (for each agent)
while True:
    actions = np.random.randn(num_agents, action_size) # select an action (for each agent)
    actions = np.clip(actions, -1, 1)                  # all actions between -1 and 1
    environment_info = environment.step(actions)[brain_name]           # send all actions to tne environment
    next_states = environment_info.vector_observations         # get next state (for each agent)
    rewards = environment_info.rewards                         # get reward (for each agent)
    dones = environment_info.local_done                        # see if episode finished
    scores += environment_info.rewards                         # update the score (for each agent)
    states = next_states                               # roll over states to next time step
    if np.any(dones):                                  # exit loop if episode finished
        break
print('Total score (averaged over agents) this episode: {}'.format(np.mean(scores)))

environment.close()
