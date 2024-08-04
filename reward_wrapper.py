import gymnasium as gym
from gymnasium import spaces

class RewardWrapper(gym.Wrapper):
    def __init__(self, env):
        super(RewardWrapper, self).__init__(env)

    def step(self, action):
        observation, reward, done, truncated, info = self.env.step(action)
        if done and reward == 1.0:
            reward = 10
        else:
            reward = -1
        return observation, reward, done, truncated, info

# Wrapping the FrozenLake environment
env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=True)
wrapped_env = RewardWrapper(env)

# Train the agent using the wrapped environment
from stable_baselines3 import DQN, PPO

dqn_model_wrapped = DQN('MlpPolicy', wrapped_env, verbose=1)
dqn_model_wrapped.learn(total_timesteps=100000)
dqn_model_wrapped.save("dqn_frozenlake_wrapped")

ppo_model_wrapped = PPO('MlpPolicy', wrapped_env, verbose=1)
ppo_model_wrapped.learn(total_timesteps=100000)
ppo_model_wrapped.save("ppo_frozenlake_wrapped")
