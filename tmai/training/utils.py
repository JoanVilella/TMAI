from dataclasses import dataclass
from collections import deque
import numpy as np
import random
from typing import Generic, Iterable, TypeVar

import sys
sys.path.append("C:/Users/jvile/Desktop/TFG/TMAI")

from tmai.agents.agent import Agent, RandomGamepadAgent
from tmai.env.TMNFEnv import TrackmaniaEnv

T = TypeVar("T")


@dataclass
class Transition:
    state: np.ndarray
    action: np.ndarray
    next_state: np.ndarray
    reward: float
    done: bool


Episode = list[Transition]


def total_reward(episode: Episode) -> float:
    return sum([t.reward for t in episode])


class Buffer(Generic[T]):
    def __init__(self, capacity=100000):
        self.capacity = capacity
        self.memory = deque([], maxlen=capacity)

    def append(self, obj: T):
        self.memory.append(obj)

    def append_multiple(self, obj_list: list[T]):
        for obj in obj_list:
            self.memory.append(obj)

    def sample(self, batch_size) -> Iterable[T]:
        return random.sample(self.memory, batch_size)

    def reset(self):
        self.memory.clear()

    def __len__(self):
        return len(self.memory)


class TransitionBuffer(Buffer[Transition]):
    def __init__(self, capacity=100000):
        super().__init__(capacity)

    def append_episode(self, episode: Episode):
        self.append_multiple(episode)

    def get_batch(self, batch_size):
        batch_of_transitions = self.sample(batch_size)
        states = np.array([t.state for t in batch_of_transitions])
        actions = np.array([t.action for t in batch_of_transitions])
        next_states = np.array([t.next_state for t in batch_of_transitions])
        rewards = np.array([t.reward for t in batch_of_transitions])
        dones = np.array([t.done for t in batch_of_transitions])

        return Transition(states, actions, next_states, rewards, dones)


def play_episode(
    agent: Agent, env: TrackmaniaEnv, render=False, act_value=None
) -> Episode:
    # Lista para almacenar las transiciones del episodio
    episode = []
    
    # Reinicia el entorno y obtiene la primera observación
    observation = env.reset()
    
    # Bandera que indica si el episodio ha terminado
    done = False
    
    # Contador de pasos en el episodio
    step = 0
    
    # Bucle principal que simula un episodio hasta que termine
    while not done:
        # Guarda la observación anterior
        prev_obs = observation
        
        # Determina la acción a tomar, ya sea mediante el agente o una función de valor externa
        if act_value is not None:
            action = act_value()
        else:
            action = agent.act(observation)
        
        # Imprime la acción tomada (para propósitos de depuración)
        print(action)
        
        # Ejecuta la acción en el entorno y obtiene la nueva observación, recompensa y estado de finalización
        observation, reward, done, info = env.step(action)
        
        # Crea una transición con la información del paso actual
        transition = Transition(prev_obs, action, observation, reward, done)
        
        # Agrega la transición a la lista del episodio
        episode.append(transition)
        
        # Incrementa el contador de pasos
        step += 1
        
        # Si se especifica, muestra visualmente el entorno en cada paso
        if render:
            env.render()

    # Devuelve la lista de transiciones que forman el episodio
    return episode



if __name__ == "__main__":
    env = TrackmaniaEnv(action_space="gamepad")
    agent = RandomGamepadAgent()
    while True:
        episode = play_episode(agent, env)
