import sys
import gym
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import DQN

# Create and wrap the environment
env = gym.make('LunarLander-v2')
env = DummyVecEnv([lambda: env])

# Load the trained agent
model = DQN.load(sys.argv[1])

# Enjoy trained agent
obs = env.reset()
for i in range(1000):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()
