import gymnasium as gym
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback

# Create the FrozenLake environment
env = gym.make('FrozenLake-v1', map_name='4x4', is_slippery=True)

# DQN Training
dqn_model = DQN('MlpPolicy', env, verbose=1, tensorboard_log="./dqn_frozenlake_tensorboard/")
dqn_model.learn(total_timesteps=100000, log_interval=4)
dqn_model.save("dqn_frozenlake")

# PPO Training
ppo_model = PPO('MlpPolicy', env, verbose=1, tensorboard_log="./ppo_frozenlake_tensorboard/")
ppo_model.learn(total_timesteps=100000, log_interval=4)
ppo_model.save("ppo_frozenlake")
