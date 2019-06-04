from pyvirtualdisplay import Display
import gym
import gym_duckietown
from wrappers import NormalizeWrapper, ImgWrapper, DtRewardWrapper, ActionWrapper, ResizeWrapper
import logging
import sys
from stable_baselines.ddpg.policies import *
from stable_baselines.common.vec_env import *
from stable_baselines.common.atari_wrappers import *
from stable_baselines import DDPG
from stable_baselines.common import set_global_seeds
from stable_baselines.ddpg.noise import *
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

def make_env(env_id, rank, seed=0):
    """
    Utility function for multiprocessed env.

    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environments you wish to have in subprocesses
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    """
    def _init():

        return env
    set_global_seeds(seed)
    return _init

# Create the environment.
env_id = "Duckietown-loop_pedestrians-v0"
env = gym.make(env_id)
#env = WarpFrame(env)
env = ResizeWrapper(env) # Resize to 120, 160, 3
env = NormalizeWrapper(env)
#env = ImgWrapper(env) # to make the images from 160x120x3 into 3x160x120
env = ActionWrapper(env)
env = DtRewardWrapper(env)
env = DummyVecEnv([make_env(env_id, i) for i in range(1)])
env = VecFrameStack(env, n_stack=4)

# Model parameters.
total_timesteps = 1e6
model_type = "ddpg"
time_stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M")

#parser.add_argument("--seed", default=0, type=int)  # Sets Gym, PyTorch and Numpy seeds
#parser.add_argument("--start_timesteps", default=1e4, type=int)  # How many time steps purely random policy is run for
#parser.add_argument("--eval_freq", default=5e3, type=float)  # How often (time steps) we evaluate
#parser.add_argument("--expl_noise", default=0.1, type=float)  # Std of Gaussian exploration noise
#parser.add_argument("--policy_noise", default=0.2, type=float)  # Noise added to target policy during critic update
#parser.add_argument("--noise_clip", default=0.5, type=float)  # Range to clip target policy noise
#parser.add_argument("--policy_freq", default=2, type=int)  # Frequency of delayed policy updates
#parser.add_argument("--env_timesteps", default=500, type=int)  # Frequency of delayed policy updates


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
    model = DDPG.load(sys.argv[1], env=env, tensorboard_log=tensorboard_log)

    # Properly log timesteps.
    reset_num_timesteps = False

# Create a new model.
else:
    print("Training a model from scratch...")
    model_name = f"{time_stamp}-{model_type}_duckietown_{total_timesteps}"

    n_actions = env.action_space.shape[-1]
    param_noise = None
    action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=float(0.5) * np.ones(n_actions))

    # Create the model and attach tensorboard.
    tensorboard_log = f"logs/{model_name}"
    model = DDPG(
        policy=CnnPolicy,
        env=env,
        param_noise=param_noise,
        action_noise=action_noise,
        gamma=0.99, # as in repo
        tau=0.005, # as in repo
        batch_size=32, # as in repo
        memory_limit=10000, # as in repo
        verbose=1,
        tensorboard_log=tensorboard_log
        )

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
