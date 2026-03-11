# FrozenLake RL Agent ❄️🤖

An implementation of Reinforcement Learning agents in the **FrozenLake-v1** environment (OpenAI Gym/Farama Foundation) using **Stable-Baselines3** and a **Flask** interaction API.

## 📌 Overview
This project focuses on training and comparing two core Reinforcement Learning algorithms: **DQN** (Deep Q-Network) and **PPO** (Proximal Policy Optimization). It includes a custom Reward Wrapper designed to improve convergence and handle the sparse reward nature of the FrozenLake environment.

## 🛠️ Tech Stack
* **Environment:** Gymnasium `FrozenLake-v1` (4x4)
* **RL Framework:** [Stable-Baselines3](https://stable-baselines3.readthedocs.io/)
* **Backend API:** Flask (for real-time agent interaction)
* **Monitoring:** TensorBoard

## 🚀 Environment & Algorithm Selection
The agents were trained under two different reward structures:
1.  **Default Reward:** The standard environment reward (1 for reaching the goal, 0 otherwise).
2.  **Custom Reward Wrapper:** A modified structure to encourage exploration and penalize falling into "holes," accelerating the learning process.

### Evaluation Results (100 Episodes)
| Algorithm | Reward Wrapper | Average Reward | Success Rate |
| :--- | :--- | :--- | :--- |
| **DQN** | Original | X | Y% |
| **DQN** | Custom | X' | Y'% |
| **PPO** | Original | A | B% |
| **PPO** | Custom | A' | B'% |

> **Conclusion:** Based on the results, **[Algorithm Name]** performed better in this environment due to **[e.g., better handling of stochasticity / faster convergence with the custom wrapper]**.

## 💻 Instructions

### 1. Installation
Clone the repository and install the necessary dependencies:
```bash
git clone [https://github.com/your-username/frozenlake-rl.git](https://github.com/your-username/frozenlake-rl.git)
cd frozenlake-rl
pip install -r requirements.txt
