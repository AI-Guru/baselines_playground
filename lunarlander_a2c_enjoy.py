import gym

from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import A2C

# Create and wrap the environment
env = gym.make('LunarLander-v2')
env = DummyVecEnv([lambda: env])

# Load the trained agent
model = A2C.load("lunarlander-a2c")

# Enjoy trained agent
obs = env.reset()
for i in range(1000):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()
