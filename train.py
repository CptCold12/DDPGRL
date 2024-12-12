# -*- coding: utf-8 -*-
"""
Created on Sat Dec  5 17:09:54 2020

@author: Javier Escribano

Adapted on Sat Dec  7 2024 by Cameron Briginshaw
"""
from unity_env import init_environment
import numpy as np
import torch
from collections import deque
import matplotlib.pyplot as plt

from agent import DDPG

# Unity env executable path
UNITY_EXE_PATH = r'C:\Users\camsa\DDPGRLC\reacher-ddpg\Tennis_Windows_x86_64\Tennis.exe'
# Environment Goal
GOAL = 1
# Averaged score
SCORE_AVERAGED = 100
# Let us know the progress each 10 timesteps
PRINT_EVERY = 10
# Number of episode for training
N_EPISODES = 2000
# Max Timesteps
MAX_TIMESTEPS = 1000

# Init the reacher environment and get agents, state and action info
env, brain_name, n_agents, state_size, action_size = init_environment(UNITY_EXE_PATH)
agent = DDPG(state_size=state_size, action_size=action_size, random_seed=89)

def plot_metrics(scores, actor_grad_norms, action_entropies, action_magnitudes, success_rates):
    metrics = {
        "Scores": scores,

        "Actor Gradient Norm": actor_grad_norms,
        "Action Entropies": action_entropies,
        "Action Magnitudes": action_magnitudes,
        "Successes Per Episode": success_rates,
    }

    for metric_name, metric_values in metrics.items():
        plt.figure()
        plt.plot(metric_values)
        plt.xlabel("Episodes")
        plt.ylabel(metric_name)
        plt.savefig(f"{metric_name.replace(' ', '_').lower()}.png")
        plt.show()

def compute_gradient_norm(parameters):
    total_norm = 0.0
    for param in parameters:
        if param.grad is not None:
            param_norm = param.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    return total_norm ** 0.5

#  Method for training the agent
def train(n_episodes=N_EPISODES):
    scores_deque = deque(maxlen=SCORE_AVERAGED)
    global_scores = []
    averaged_scores = []
    actor_gradient_norms = []
    action_entropies = []
    action_magnitudes = []
    success_rates = []
  
    for episode in range(1, N_EPISODES + 1):
        # Initialize per-episode metrics
        episode_gradients = []
        episode_entropies = []
        episode_magnitudes = []
        success_count = 0

        # Get the current states for each agent
        states = env.reset(train_mode=True)[brain_name].vector_observations
        # Init the score of each agent to zeros
        scores = np.zeros(n_agents)

        for t in range(MAX_TIMESTEPS):
            actions = agent.act(states)

            # Calculate metrics
            action_entropy = -np.sum(actions * np.log(np.clip(actions, 1e-10, 1.0))) / len(actions)
            action_magnitude = np.mean(np.abs(actions))
            episode_entropies.append(action_entropy)
            episode_magnitudes.append(action_magnitude)

            # Append actor gradient norm for each timestep
            episode_gradients.append(compute_gradient_norm(agent.actor_regular.parameters()))

            env_info = env.step(actions)[brain_name]
            next_states = env_info.vector_observations
            rewards = env_info.rewards
            dones = env_info.local_done
            agent.step(states, actions, rewards, next_states, dones, t)
            states = next_states
            scores += rewards
            
            for x in rewards:
                if x > 0:
                    success_count +=1

            if np.any(dones):
                break
        
        # Calculate scores and averages
        score = np.mean(scores)
        scores_deque.append(score)
        avg_score = np.mean(scores_deque)

        global_scores.append(score)
        averaged_scores.append(avg_score)

        # Calculate and store per-episode metrics
        action_entropies.append(np.mean(episode_entropies))
        action_magnitudes.append(np.mean(episode_magnitudes))
        actor_gradient_norms.append(np.mean(episode_gradients))
        success_rates.append(success_count)

        if episode % PRINT_EVERY == 0:
            print(f'Episode {episode}\tAverage Score: {avg_score:.2f}\tEntropy: {np.mean(action_entropies):.4f}\tMagnitude: {np.mean(action_magnitudes):.4f}')
            
        if avg_score >= GOAL:  
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(episode, avg_score))
            torch.save(agent.actor_regular.state_dict(), 'actor_theta.pth')
            torch.save(agent.critic_regular.state_dict(), 'critic_theta.pth')
            break

    plot_metrics(global_scores, actor_gradient_norms, action_entropies, action_magnitudes, success_rates)
            
    return global_scores, averaged_scores

# Train the agent and get the results
scores, averages = train()

env.close()
