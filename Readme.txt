# FrozenLake RL Agent

## Overview

This project involves training and evaluating reinforcement learning agents on the FrozenLake-v1 environment using DQN and PPO algorithms from stable-baselines3. Additionally, a custom reward wrapper was implemented to modify the reward structure.

## Environment and Algorithm Selection

- **Environment**: FrozenLake-v1 (map_name='4x4')
- **Algorithms**: DQN and PPO

## Training

The agents were trained using the default reward structure and a custom reward wrapper. The training curves were saved using TensorBoard.

## Evaluation

The agents were evaluated by interacting with the environment through a Flask API. The average reward and success rate over 100 episodes were calculated.

## Results

- **DQN**: Average Reward: X, Success Rate: Y
- **PPO**: Average Reward: A, Success Rate: B

## Reward Wrapper

A custom reward wrapper was implemented to modify the reward structure. The agents were retrained and the results compared to the original reward structure.

## Conclusion

Based on the results, [algorithm] performed better for this environment due to [reasons].

## Instructions

1. Install dependencies: `pip install -r requirements.txt`
2. Train the agents: `python train_agents.py`
3. Start the API server: `python FrozenLakeAPI_structure.py`
4. Evaluate the agents
