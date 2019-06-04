import sys

import gym
import ale

from stable_baselines import *
from stable_baselines.deepq.policies import *
from stable_baselines.common.vec_env import *
from stable_baselines.common.atari_wrappers import *

class FlappyBirdRewardWrapper(gym.RewardWrapper):
    def __init__(self, env):
        super(FlappyBirdRewardWrapper, self).__init__(env)

    def reward(self, reward):
        return np.clip(reward, -1., 1.)

env_id = "SpaceInvaders-v0"
env = gym.make(env_id)
env = WarpFrame(env)
env = ScaledFloatFrame(env)
env = FlappyBirdRewardWrapper(env)
env = DummyVecEnv([lambda:env])
env = VecFrameStack(env, n_stack=4)

# Load the trained agent
model = DQN.load(sys.argv[1])

# Enjoy trained agent
obs = env.reset()
for i in range(1000):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()
