import requests
from stable_baselines3 import DQN, PPO

# Load the pre-trained models
dqn_model = DQN.load("dqn_frozenlake")
ppo_model = PPO.load("ppo_frozenlake")

def evaluate_agent(model, num_episodes=100):
    rewards = []
    successes = 0

    for _ in range(num_episodes):
        response = requests.post('http://localhost:5005/new_game')
        env_id = response.json()['env_id']
        
        response = requests.post('http://localhost:5005/reset', json={'env_id': env_id})
        observation = response.json()['observation']
        
        done = False
        total_reward = 0
        
        while not done:
            action, _ = model.predict(observation)
            response = requests.post('http://localhost:5005/step', json={'env_id': env_id, 'action': int(action)})
            result = response.json()
            observation = result['observation']
            reward = result['reward']
            done = result['done']
            total_reward += reward
        
        rewards.append(total_reward)
        if total_reward == 1.0:
            successes += 1
    
    average_reward = sum(rewards) / len(rewards)
    success_rate = successes / num_episodes
    
    return average_reward, success_rate

if __name__ == "__main__":
    avg_reward_dqn, success_rate_dqn = evaluate_agent(dqn_model)
    print(f"DQN - Average Reward: {avg_reward_dqn}, Success Rate: {success_rate_dqn}")

    avg_reward_ppo, success_rate_ppo = evaluate_agent(ppo_model)
    print(f"PPO - Average Reward: {avg_reward_ppo}, Success Rate: {success_rate_ppo}")

