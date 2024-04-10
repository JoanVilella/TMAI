import sys
sys.path.append("C:/Users/jvile/Desktop/TFG/TMAI")

import torch
import torch.optim as optim
import torch.nn as nn

from tmai.training.utils import Buffer, Transition, play_episode
from tmai.env.TMNFEnv import TrackmaniaEnv
from tmai.agents.DQN_agent import EpsilonGreedyDQN
import numpy as np

class DQNTester:
    def __init__(self, batch_size=32, N_epochs=100):
        self.N_epochs = N_epochs
        self.target_update = 2
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.buffer = Buffer(capacity=10000)
        self.env = TrackmaniaEnv(action_space="arrows")
        self.agent = EpsilonGreedyDQN(
            input_size=self.env.observation_space.shape[0], device=self.device
        )
        self.optimizer = optim.Adam(self.agent.policy.parameters(), lr=0.001)
        self.loss = nn.SmoothL1Loss()

    def test(self, model_dir, policy_model_name, target_model_name):
        cumulative_reward_list = []
        episode_length_list = []

        # Cargar el modelo preentrenado
        self.agent.load_model(model_dir, policy_model_name, target_model_name)

        for epoch in range(self.N_epochs):
            self.env.reset()
            episode = []
            observation = self.env.reset()
            done = False
            time = 0

            while not done:
                prev_obs = observation
                # Utilizar el modelo preentrenado para tomar acciones
                action = self.agent.act(observation)
                action[0] = 1
                action[1] = 0
                observation, reward, done, time = self.env.step(action)
                transition = Transition(prev_obs, action, observation, reward, done)
                episode.append(transition)
                step += 1
                # self.env.render()
                # Omitir el paso de optimización, ya que no estamos entrenando aquí

            # Guardar la recompensa acumulada del episodio
            cumulative_reward = sum([transition.reward for transition in episode])
            cumulative_reward_list.append(cumulative_reward)
            # Guardar la duración del episodio
            episode_length_list.append(time)

            # Guardar las métricas en un archivo
            # np.save('cumulative_reward_list.npy', cumulative_reward_list)
            # np.save('episode_length_list.npy', episode_length_list)

            # Añadir el episodio al buffer de experiencia
            self.buffer.append_multiple(episode)

            print(f"Testing: Epoch {epoch}")

        print("Testing finished")

if __name__ == "__main__":
    tester = DQNTester(N_epochs=10)
    tester.test("C:/Users/jvile/Desktop/TFG/TMAI/models", "policy_model_20240410110334.pth", "target_model_20240410110334.pth")



