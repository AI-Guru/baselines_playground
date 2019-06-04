import sys

import gym
import gym_duckietown
from gym_duckietown.envs.duckietown_env import DuckietownEnv

from stable_baselines.common.vec_env import *
from stable_baselines import *
from stable_baselines.common.atari_wrappers import *

def get_action_meanings(self):
    print("!!!")
    return {}


DuckietownEnv.get_action_meanings = get_action_meanings

# Create and wrap the environment
env = gym.make("Duckietown-loop_pedestrians-v0")
#env = wrap_deepmind(env, episode_life=False, clip_rewards=True, frame_stack=True, scale=True)
env = WarpFrame(env)
env = ScaledFloatFrame(env)
#env = ClipRewardEnv(env)
env = DummyVecEnv([lambda: env])
env = SubprocVecEnv([lambda: env])

env = VecFrameStack(env, n_stack=4)

# Load the trained agent
model = A2C.load(sys.argv[1])

# Enjoy trained agent
obs = env.reset()
for i in range(1000):
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)
    if done == True:
        env.reset()
        continue
    env.render()
