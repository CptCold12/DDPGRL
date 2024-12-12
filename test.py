# -*- coding: utf-8 -*-
from unity_env import init_environment
import numpy as np
import torch
from agent import DDPG

# Unity env executable path
#UNITY_EXE_PATH = r'C:\Users\camsa\DDPGRLC\reacher-ddpg\Reacher_Windows_x86_64\Reacher.exe'
UNITY_EXE_PATH = r'C:\Users\camsa\DDPGRLC\reacher-ddpg\Tennis_Windows_x86_64\Tennis.exe'
# Init the reacher environment and get agents, state and action info
env, brain_name, n_agents, state_size, action_size = init_environment(UNITY_EXE_PATH)
agent = DDPG(state_size=state_size, action_size=action_size, random_seed=29)

# Load trained weights
agent.critic_regular.load_state_dict(torch.load('critic_theta.pth', map_location=torch.device('cpu')))
agent.actor_regular.load_state_dict(torch.load('actor_theta.pth', map_location=torch.device('cpu')))


env_info = env.reset(train_mode=False)[brain_name]     
states = env_info.vector_observations                  
scores = np.zeros(n_agents)   
            
while True:
    actions = agent.act(states)                     
    env_info = env.step(actions)[brain_name]          
    next_states = env_info.vector_observations       
    rewards = env_info.rewards                        
    dones = env_info.local_done                        
    scores += env_info.rewards                         
    states = next_states                              
    if np.any(dones):                                  
        break

env.close()