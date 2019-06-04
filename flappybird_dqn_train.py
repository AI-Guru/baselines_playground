import gym
import gym_ple

from stable_baselines import *
from stable_baselines.deepq.policies import *
from stable_baselines.common.vec_env import *
from stable_baselines.common.atari_wrappers import *

class FlappyBirdRewardWrapper(gym.RewardWrapper):
    def __init__(self, env):
        super(FlappyBirdRewardWrapper, self).__init__(env)

    def reward(self, reward):
        return np.clip(reward, -1., 1.)

env_id = "FlappyBird-v0"
env = gym.make(env_id)
env = WarpFrame(env)
env = ScaledFloatFrame(env)
env = FlappyBirdRewardWrapper(env)
env = DummyVecEnv([lambda:env])
env = VecFrameStack(env, n_stack=4)

total_timesteps = 1000000
model_type = "dqn"

model_name = "{}_{}_{}".format(model_type, env_id, total_timesteps)
print("Training model {}...".format(model_name))

tensorboard_log = "logs/{}".format(model_name)
print("Logging to {}...".format(tensorboard_log))

model = DQN(
    CnnPolicy,
    env=env,
    learning_rate=0.00025,
    target_network_update_freq=10000,
    learning_starts=50000,
    buffer_size=1000000,
    verbose=1, tensorboard_log=tensorboard_log)

model.learn(total_timesteps=total_timesteps)
model.save(model_name)
