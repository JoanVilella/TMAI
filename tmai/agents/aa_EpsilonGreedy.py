import os
from datetime import datetime
import numpy as np
import torch
from tmai.agents.agent import Agent
from tmai.agents.aa_DQN_base import DQNBase
from tmai.agents.Linear_DQN import LinearDQN
from tmai.agents.aa_Linear_DQN import ConvDQN



class EpsilonGreedyDQN(Agent):
    def __init__(self, input_size, device, model_cls=DQNBase, eps=1e-3):
        super().__init__()
        self.device = device
        self.eps_start = 0.9
        self.eps_end = eps
        self.eps_decay = 200000
        self.action_correspondance = {
            i + 2 * j + 4 * k + 8 * l: [i, j, k, l]
            for i in range(2)
            for j in range(2)
            for k in range(2)
            for l in range(2)
        }

        print(len(self.action_correspondance))
        self.policy = model_cls(input_size, len(self.action_correspondance))
        self.target = model_cls(input_size, len(self.action_correspondance))
        self.policy.to(self.device)
        self.target.to(self.device)
        self.step = 0

    def epsilon(self):
        epsilon = self.eps_end + (self.eps_start - self.eps_end) * np.exp(
            -1.0 * self.step / self.eps_decay
        )
        if epsilon < 0.05:
            epsilon = 0.05
        return epsilon

    def act(self, observation):
        epsilon = self.epsilon()
        option = np.random.rand() < epsilon
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

    input_size = 17  # or whatever your input size is
    device = "cpu"
    agent_linear = EpsilonGreedyDQN(input_size, device, model_cls=LinearDQN)
    agent_conv = EpsilonGreedyDQN(input_size, device, model_cls=ConvDQN)

    import matplotlib.pyplot as plt

    epsilon_values_linear = []
    epsilon_values_conv = []

    for step in range(200000):
        agent_linear.step = step
        epsilon_values_linear.append(agent_linear.epsilon())

        agent_conv.step = step
        epsilon_values_conv.append(agent_conv.epsilon())

    plt.plot(epsilon_values_linear, label='Linear DQN')
    plt.plot(epsilon_values_conv, label='Conv DQN')
    plt.xlabel('Step')
    plt.ylabel('Epsilon')
    plt.title('Epsilon Decay')
    plt.legend()
    plt.show()

