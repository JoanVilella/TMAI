from stable_baselines3 import PPO

import sys
sys.path.append("C:/Users/jvile/Desktop/TFG/TMAI")


from tmai.env.TMNFEnv import TrackmaniaEnv

if __name__ == "__main__":
    env = TrackmaniaEnv(action_space="gamepad")
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=10000)