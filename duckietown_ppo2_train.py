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

# Create a virtual display for running headless.
if "virtualdisplay" in sys.argv:
    print("Using virtual display...")
    display = Display(visible=0, size=(1400, 900))
    display.start()
else:
    print("Using real display...")
    display = None

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
        return env
    set_global_seeds(seed)
    return _init

# Create the environment.
env_id = "Duckietown-loop_pedestrians-v0"
if "multiprocessing" not in sys.argv:
    print("Not using multiprocessing...")
    env = DummyVecEnv([make_env(env_id, i) for i in range(1)])
    env = VecFrameStack(env, n_stack=4)
else:
    cpu_count = multiprocessing.cpu_count()
    print("Using multiprocessing on {} CPUs...".format(cpu_count))
    env = SubprocVecEnv([make_env(env_id, i) for i in range(cpu_count)])
    env = VecFrameStack(env, n_stack=4)

# Model parameters.
total_timesteps = 1250000
model_type = "ppo2"
time_stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M")

# Load an old model an prepare it for retraining.
if len(sys.argv) != 1:
    print("Loading model and continue training...")

    # Getting the new model name from the old.
    old_model_name = sys.argv[1].split("/")[-1].split(".")[0]
    old_timesteps = int(old_model_name.split("_")[-1])
    model_name = f"{time_stamp}-{model_type}_duckietown_{total_timesteps + old_timesteps}"
    print("Using model {}".format(old_model_name))

    # Load the model and attach tensorboard.
    tensorboard_log = f"logs/{model_name}"
    model = PPO2.load(sys.argv[1], env=env, tensorboard_log=tensorboard_log)

    # Properly log timesteps.
    reset_num_timesteps = False

# Create a new model.
else:
    print("Training a model from scratch...")
    model_name = f"{time_stamp}-{model_type}_duckietown_{total_timesteps}"

    # Create the model and attach tensorboard.
    tensorboard_log = f"logs/{model_name}"
    model = PPO2(CnnPolicy, env, verbose=1, tensorboard_log=tensorboard_log)

    # Properly log timesteps.
    reset_num_timesteps = True

# Train the model.
print(f"Training model {model_name}...")
print(f"Logging to {tensorboard_log}...")
model.learn(total_timesteps=total_timesteps, reset_num_timesteps=reset_num_timesteps)
model.save(model_name)

# Close display.
if display != None:
    display.popen.terminate()
