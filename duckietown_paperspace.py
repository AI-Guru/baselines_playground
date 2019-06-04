from pyvirtualdisplay import Display
import gym
import gym_duckietown
import logging
import sys
from stable_baselines.common.policies import *
from stable_baselines.common.vec_env import *
from stable_baselines.common.atari_wrappers import *
from stable_baselines import *
from stable_baselines.common import set_global_seeds
import multiprocessing
import datetime

logging.disable(sys.maxsize)
logger = logging.getLogger()
logger.disabled = True
logger = logging.getLogger("gym-duckietown")
logger.disabled = True
logging.disable(sys.maxsize)
logging.basicConfig(level="ERROR")

display = Display(visible=0, size=(1400, 900))
display.start()

env_id = "Duckietown-loop_pedestrians-v0"

class DtRewardWrapper(gym.RewardWrapper):
    def __init__(self, env):
        super(DtRewardWrapper, self).__init__(env)

    def reward(self, reward):
        if reward == -1000:
            reward = -10
        elif reward > 0:
            reward += 10
        else:
            reward += 4

        return reward

def make_env(env_id, rank, seed=0):
    """
    Utility function for multiprocessed env.

    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environments you wish to have in subprocesses
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    """
    def _init():
        env = gym.make(env_id)
        env.seed(seed + rank)
        env = WarpFrame(env)
        env = ScaledFloatFrame(env)
        env = DtRewardWrapper(env)
        #env = ClipRewardEnv(env)
        return env
    set_global_seeds(seed)
    return _init

env = DummyVecEnv([make_env(env_id, i) for i in range(1)])
env = VecFrameStack(env, n_stack=4)

# Multiprocessing.
#env = SubprocVecEnv([make_env(env_id, i) for i in range(multiprocessing.cpu_count())])
#env = VecFrameStack(env, n_stack=4)


total_timesteps = 2500000
model_types = ["ppo2", "a2c"]
for model_type in model_types:
    time_stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M")
    model_name = f"{time_stamp}-{model_type}_duckietown_{total_timesteps}"
    print(f"Training model {model_name}...")

    tensorboard_log = f"logs/{model_name}"
    print(f"Logging to {tensorboard_log}...")

    if model_type == "a2c":
        model = A2C(CnnPolicy, env, verbose=1, tensorboard_log=tensorboard_log)
    elif model_type == "ppo2":
        model = PPO2(CnnPolicy, env, verbose=1, tensorboard_log=tensorboard_log)

    model.learn(total_timesteps=total_timesteps)
    model.save(model_name)

display.popen.terminate()
