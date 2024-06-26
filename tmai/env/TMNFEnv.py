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
        self.max_steps = 120 # Max steps per episode 1000 = 8min and 20 seconds TODO 1000
        self.command_frequency = 50
        self.last_action = None
        self.low_speed_steps = 0
        self.contact_steps = 0

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
            if self.n_steps >= self.max_steps or self.total_reward < -300 #TODO 300
            else False
        )
        self.total_reward += self.reward
        self.n_steps += 1
        info = self.state.time
        time.sleep(self.command_frequency * 10e-3)
        return self.observation, self.reward, done, info

    def reset(self):
        # print("reset")
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

    def render_return(self):
        return self.total_reward, self.speed, self.state.time
    

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
    def has_lateral_contact(self):
        return self.state.scene_mobil.has_any_lateral_contact
    
    @property
    def reward(self):

        reward = self.speed

        if self.speed < 5:
            reward -= 100

        return reward


    """
    @property
    def reward(self):
        reward = self.speed
        
        # Penalizar si la velocidad es muy baja
        low_speed_penalty = 0
        if self.speed < 5:
            self.low_speed_steps += 1
            if self.low_speed_steps > 6: # 3 seconds aprox
                low_speed_penalty -= 1000  # Penalizar fuertemente si ha estado en baja velocidad por más de 6 pasos
        else:
            self.low_speed_steps = 0  # Reiniciar contador si la velocidad es adecuada
        
        # Penalización por contacto lateral
        max_contact_steps = 1  # Definir el máximo de pasos permitidos en contacto lateral
        contact_penalty = 0
        if self.has_lateral_contact:
            self.contact_steps += 1
            if self.contact_steps > max_contact_steps:
                contact_penalty -= 5 * (self.contact_steps - max_contact_steps)  # Penalizar proporcionalmente
        else:
            self.contact_steps = 0  # Reiniciar contador si no hay contacto lateral

        total_reward = reward + contact_penalty + low_speed_penalty
        
        # Imprimir la recompensa por velocidad y las penalizaciones
        # print(f"Speed Reward: {reward}, Contact Penalty: {contact_penalty}, Low Speed Penalty: {low_speed_penalty}, Total Reward: {total_reward}")
        
        return total_reward
    """

    
    """
    @property
    def reward(self):
        reward = 0
    
        # Penalizar movimientos de volcado excesivos
        roll_penalty = -abs(self.state.yaw_pitch_roll[2]) / 3.15
        reward += roll_penalty
        
        # Penalizar velocidad baja o alta
        if 10 < self.speed < 50:  # Velocidad objetivo entre 10 y 50
            # Penalizar si la velocidad está fuera del rango objetivo
            speed_penalty = -abs(30 - self.speed)  # Penalización proporcional a la distancia de la velocidad actual al objetivo (30)
            reward += speed_penalty
        else:
            # Penalizar fuertemente si la velocidad es demasiado baja o demasiado alta
            reward -= 50
        
        # Penalizar si la observación está muy cercana al suelo
        if sum(self.observation[:-1]) < 0.06:
            # Penalización adicional si la observación es muy cercana al suelo
            reward -= 50
        
        # Penalizar si el pitch está mirando hacia arriba (pitch positivo)
        pitch_penalty = -abs(self.state.yaw_pitch_roll[1]) / 3.15
        reward += pitch_penalty
        
        # Recompensa constante para mantener un incentivo constante
        constant_reward = -0.3
        reward += constant_reward

        return reward

    """
    
    """
    @property
    def reward(self):
        speed = self.speed
        speed_reward = speed / 400  # La recompensa es proporcional a la velocidad y se normaliza dividiendo por 400 (valor maximo)

        # Constant reward
        constant_reward = -0.3

        # Calcula una penalización basada en el ángulo de inclinación del vehículo (si se pone boca abajo, se penaliza)
        roll_reward = -abs(self.state.yaw_pitch_roll[2]) / 3.15

        # Calcula una penalización basada en pitch, por si se levanta mucho la nariz
        pitch_reward = -abs(self.state.yaw_pitch_roll[1]) / 3.15

        # Si el valor de todos los rayos es 0 (no hay obstáculos), se aplica una penalización adicional de -100
        if sum(self.observation[:-1]) < 0.06:
            constant_reward = -1

        # Print speed reward and roll reward
        # print(f"speed reward: {speed_reward}, roll reward: {roll_reward}, self.observation: {self.observation}")



        return speed_reward + roll_reward + constant_reward + pitch_reward

    """
    
    """
    @property
    def reward(self):

        print(f"display_speed: {self.state.display_speed}")
        print(f"input_accelerate: {self.state.input_accelerate}")
        print(f"input_brake: {self.state.input_brake}")
        print(f"input_gas: {self.state.input_gas}")
        print(f"input_left: {self.state.input_left}")
        print(f"input_right: {self.state.input_right}")
        print(f"input_steer: {self.state.input_steer}")
        print(f"position: {self.state.position}")
        print(f"race_time: {self.state.race_time}")
        print(f"rotation_matrix: {self.state.rotation_matrix}")
        print(f"time: {self.state.time}")
        print(f"velocity: {self.state.velocity}")
        print(f"yaw_pitch_roll: {self.state.yaw_pitch_roll}")

        return self.speed

    """

    """
    @property
    def reward(self):

        speed = self.speed
        speed_reward = speed

        constant_reward = -0.3

        roll_reward = -abs(self.state.yaw_pitch_roll[2]) / 3.15
        
        if min(self.observation) < 0.06:
            # Si el valor mínimo de la observación es menor que 0.06, se aplica una penalización adicional de -100
            constant_reward -= 50
        elif 10 < speed < 30: # TODO 100
            # Si la velocidad está entre 10 y 100, se establece una recompensa constante de -1 y se anula la recompensa del gas
            speed_reward += 50 # TODO 1
        elif speed < 10 or speed > 50:
            # Si la velocidad es inferior a 10, se incrementa un contador de pasos de baja velocidad y se aplica una penalización adicional de -5 por cada paso

            speed_reward -= 50 


        return speed_reward + roll_reward + constant_reward  

    """


    """    
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
        gas_reward = self.last_action[0]  # Recompensa proporcional a la acción del gas multiplicada por 2 TODO * 2

        # Condiciones adicionales que introducen penalizaciones
        if self.last_action[0] < 0:
            # Si la acción del gas es negativa, se reduce la penalización constante y se anula la recompensa del gas
            constant_reward -= 10
            gas_reward = 0

        if min(self.observation) < 0.06:
            # Si el valor mínimo de la observación es menor que 0.06, se aplica una penalización adicional de -100
            constant_reward -= 50 # TODO 100

        elif 10 < speed < 30: # TODO 100
            # Si la velocidad está entre 10 y 100, se establece una recompensa constante de -1 y se anula la recompensa del gas
            speed_reward = -0.5 # TODO 1
            gas_reward = 0

        elif speed < 10:
            # Si la velocidad es inferior a 10, se incrementa un contador de pasos de baja velocidad y se aplica una penalización adicional de -5 por cada paso
            self.low_speed_steps += 1
            speed_reward = -5 * self.low_speed_steps
            gas_reward = 0

        else:
            # Si no se cumple ninguna de las condiciones anteriores, se reinicia el contador de pasos de baja velocidad
            self.low_speed_steps = 0        

        # Print the speed reward, roll reward, constant reward and gas reward
        print(f"speed reward: {speed_reward}, roll reward: {roll_reward}, constant reward: {constant_reward}, gas reward: {gas_reward}")


        # Devuelve la suma de todas las recompensas y penalizaciones calculadas
        return speed_reward + roll_reward + constant_reward + gas_reward
    """

