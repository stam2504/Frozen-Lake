from flask import Flask, jsonify, request
import gymnasium as gym
from uuid import uuid4
from typing import Dict, Any

class FrozenLakeAPI:
    def __init__(self) -> None:
        self.app = Flask(__name__)
        self.games = {}

    def run_server(self) -> None:
        self.app.route('/new_game', methods=['POST'])(self.new_game)
        self.app.route('/step', methods=['POST'])(self.step)
        self.app.route('/reset', methods=['POST'])(self.reset)
        self.app.run(host="localhost", port=5005)

    def new_game(self) -> Any:
        env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=True)
        env_id = str(uuid4())
        self.games[env_id] = env
        return jsonify({'env_id': env_id})

    def reset(self) -> Any:
        data = request.get_json()
        env_id = data['env_id']
        if env_id in self.games:
            observation = self.games[env_id].reset()[0]
            return jsonify({'observation': observation})
        else:
            return jsonify({'error': 'Environment ID not found'}), 400

    def step(self) -> Any:
        data = request.get_json()
        env_id = data['env_id']
        action = data['action']
        if env_id in self.games:
            observation, reward, done, truncated, info = self.games[env_id].step(action)
            return jsonify({'observation': observation, 'reward': reward, 'done': done, 'truncated': truncated, 'info': info})
        else:
            return jsonify({'error': 'Environment ID not found'}), 400

if __name__ == '__main__':
    api = FrozenLakeAPI()
    api.run_server()
