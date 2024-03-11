import time
from typing import TypeVar

import numpy as np
from gym import Env
from gym.spaces import Box, MultiBinary

from tmai.env.TMIClient import ThreadedClient
from tmai.env.utils.GameCapture import GameViewer
from tmai.env.utils.GameInteraction import (
    ArrowInput,
    GamepadInputManager,
    KeyboardInputManager,
)
from tmai.env.utils.GameLaunch import GameLauncher

import sys
sys.path.append("C:/Users/jvile/Desktop/TFG/TMAI")

ArrowsActionSpace = MultiBinary((4,))  # none up down right left
ControllerActionSpace = Box(
    low=np.array([0.0, -1.0]), high=np.array([1.0, 1.0]), shape=(2,), dtype=np.float32
)  # gas and steer
ActType = TypeVar("ActType")
ObsType = TypeVar("ObsType")


class TrackmaniaEnv(Env):
    """
    Gym env interfacing the game.
    Observations are the rays of the game viewer.
    Controls are the arrow keys or the gas and steer.
    """

    def __init__(
        self,
        action_space: str = "arrows",
        n_rays: int = 16,
    ):
        self.action_space = (
            ArrowsActionSpace if action_space == "arrows" else ControllerActionSpace
        )
        self.observation_space = Box(
            low=0.0, high=1.0, shape=(n_rays + 1,), dtype=np.float32
        )

        self.input_manager = (
            KeyboardInputManager()
            if action_space == "arrows"
            else GamepadInputManager()
        )

        game_launcher = GameLauncher()
        if not game_launcher.game_started:
            game_launcher.start_game()
            print("game started")
            input("press enter when game is ready")
            time.sleep(4)

        self.viewer = GameViewer(n_rays=n_rays)
        self.simthread = ThreadedClient()
        self.total_reward = 0.0
        self.n_steps = 0
        self.max_steps = 1000
        self.command_frequency = 50
        self.last_action = None
        self.low_speed_steps = 0

    """
    action[0] = up
    action[1] = down
    action[2] = right
    action[3] = left
    """

    def step(self, action):
        self.last_action = action
        # print(f"action: {action}") # Array with 2 values
        # plays action
        self.action_to_command(action)
        done = (
            True
            if self.n_steps >= self.max_steps or self.total_reward < -300
            else False
        )
        self.total_reward += self.reward
        self.n_steps += 1
        info = {}
        time.sleep(self.command_frequency * 10e-3)
        return self.observation, self.reward, done, info

    def reset(self):
        print("reset")
        self.total_reward = 0.0
        self.n_steps = 0
        self._restart_race()
        self.time = 0
        self.last_action = None
        self.low_speed_steps = 0
        print("reset done")

        return self.observation

    def render(self, mode="human"):
        print(f"total reward: {self.total_reward}")
        print(f"speed: {self.speed}")
        print(f"time = {self.state.time}")

    def action_to_command(self, action):
        if isinstance(self.action_space, MultiBinary):
            return self._discrete_action_to_command(action)
        elif isinstance(self.action_space, Box):
            return self._continuous_action_to_command(action)

    def _continuous_action_to_command(self, action):
        gas, steer = action
        self.input_manager.play_gas(gas)
        self.input_manager.play_steer(steer)

    def _discrete_action_to_command(self, action):
        commands = ArrowInput.from_discrete_agent_out(action)
        self.input_manager.play_inputs_no_release(commands)

    def _restart_race(self):
        if isinstance(self.input_manager, KeyboardInputManager):
            self._keyboard_restart()
        else:
            self._gamepad_restart()

    def _keyboard_restart(self):
        self.input_manager.press_key(ArrowInput.DEL)
        time.sleep(0.1)
        self.input_manager.release_key(ArrowInput.DEL)

    def _gamepad_restart(self):
        self.input_manager.press_right_shoulder()

    @property
    def state(self):
        return self.simthread.data

    @property
    def speed(self):
        return self.state.display_speed

    @property
    def observation(self):
        return np.concatenate([self.viewer.get_obs(), [self.speed / 400]]) # Distancia de los rayos y la velocidad normalizada

    @property
    def reward(self):
        # Calcula la recompensa basada en la velocidad del vehículo
        speed = self.speed
        speed_reward = speed / 400  # La recompensa es proporcional a la velocidad y se normaliza dividiendo por 400

        # Calcula una penalización basada en el ángulo de inclinación del vehículo
        roll_reward = -abs(self.state.yaw_pitch_roll[2]) / 3.15  # Penalización proporcional al ángulo de inclinación y normalizada por 3.15

        # Asigna una penalización constante independientemente de las condiciones específicas
        constant_reward = -0.3

        # Calcula una recompensa basada en la acción del gas
        gas_reward = self.last_action[0] * 2  # Recompensa proporcional a la acción del gas multiplicada por 2

        # Condiciones adicionales que introducen penalizaciones
        if self.last_action[0] < 0:
            # Si la acción del gas es negativa, se reduce la penalización constante y se anula la recompensa del gas
            constant_reward -= 10
            gas_reward = 0

        if min(self.observation) < 0.06:
            # Si el valor mínimo de la observación es menor que 0.06, se aplica una penalización adicional de -100
            constant_reward -= 100

        elif 10 < speed < 100:
            # Si la velocidad está entre 10 y 100, se establece una recompensa constante de -1 y se anula la recompensa del gas
            speed_reward = -1
            gas_reward = 0

        elif speed < 10:
            # Si la velocidad es inferior a 10, se incrementa un contador de pasos de baja velocidad y se aplica una penalización adicional de -5 por cada paso
            self.low_speed_steps += 1
            speed_reward = -5 * self.low_speed_steps
            gas_reward = 0

        else:
            # Si no se cumple ninguna de las condiciones anteriores, se reinicia el contador de pasos de baja velocidad
            self.low_speed_steps = 0

        # Devuelve la suma de todas las recompensas y penalizaciones calculadas
        return speed_reward + roll_reward + constant_reward + gas_reward

