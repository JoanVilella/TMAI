import numpy as np
import torch
import torch.nn as nn

import sys
sys.path.append("C:/Users/jvile/Desktop/TFG/TMAI")

from tmai.agents.agent import Agent
import os
from datetime import datetime

class DQN(nn.Module):
    def __init__(self, input_size, output_size, mid_size=32, p=0.5) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_size, mid_size)
        self.fc2 = nn.Linear(mid_size, output_size)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, observation):
        x = torch.Tensor(observation).to("cuda" if torch.cuda.is_available() else "cpu")
        x = self.fc1(x)
        x = nn.ReLU()(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class EpsilonGreedyDQN(Agent):
    def __init__(self, input_size, device, eps=1e-3) -> None: # eps=1e-3
        super().__init__()
        self.device = device
        self.eps_start = 0.9
        self.eps_end = eps
        self.eps_decay = 200000 # TODO 200
        self.action_correspondance = {
            i + 2 * j + 4 * k + 8 * l: [i, j, k, l]
            for i in range(2)
            for j in range(2)
            for k in range(2)
            for l in range(2)
        }
        self.policy = DQN(input_size, len(self.action_correspondance))
        self.target = DQN(input_size, len(self.action_correspondance))
        self.policy.to(self.device)
        self.target.to(self.device)
        self.step = 0


    # Se calcula epsilon de forma dinámica. Estudiar como se calcula.
    # Lo ideal sería que al inicio se diera más peso a la exploración y luego se fuera reduciendo.
    def epsilon(self):
        epsilon = self.eps_end + (self.eps_start - self.eps_end) * np.exp(
            -1.0 * self.step / self.eps_decay
        )

        if epsilon < 0.05: # TODO define a minimum epsilon
            epsilon = 0.05

        return epsilon

    def act(self, observation):
        epsilon = self.epsilon()
        # print(f"Step: {self.step}, Epsilon: {epsilon}")
        option = np.random.rand() < epsilon
        # print(f"Option: {option}")
        if option: # Explore
            self.step += 1 
            return self.action_correspondance[
                np.random.randint(0, len(self.action_correspondance)) # Explore: Random action
            ]
        self.step += 1 # Exploit
        return self.action_correspondance[
            np.argmax(self.policy(observation).detach().cpu().numpy())
        ]
    
    def save_model(self, path):
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            policy_model_name = f"policy_model_{timestamp}.pth"
            target_model_name = f"target_model_{timestamp}.pth"
            torch.save(self.policy.state_dict(), os.path.join(path, policy_model_name))
            torch.save(self.target.state_dict(), os.path.join(path, target_model_name))


    def load_model(self, path, policy_model_name="policy_model.pth", target_model_name="target_model.pth"):
        self.policy.load_state_dict(torch.load(os.path.join(path, policy_model_name)))
        self.target.load_state_dict(torch.load(os.path.join(path, target_model_name)))


if __name__ == "__main__":

    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


    input_size = 17
    device = "cpu"
    agent = EpsilonGreedyDQN(input_size, device)

    import matplotlib.pyplot as plt

    epsilon_values = []

    for step in range(200000):
        agent.step = step
        epsilon_values.append(agent.epsilon())

    plt.plot(epsilon_values)
    plt.xlabel('Step')
    plt.ylabel('Epsilon')
    plt.title('Epsilon Decay')
    plt.show()