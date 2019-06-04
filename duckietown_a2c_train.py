import gym
import gym_duckietown

from stable_baselines.common.policies import CnnPolicy
from stable_baselines.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines import A2C

import utils
tensorboard_log = "logs/duckietown-a2c"
utils.mkdir_p(tensorboard_log)

# Create and wrap the environment
env = gym.make("Duckietown-loop_pedestrians-v0")
env = DummyVecEnv([lambda: env])
env = VecFrameStack(env, n_stack=4)

# Alternatively, you can directly use:
# model = A2C('MlpPolicy', 'LunarLander-v2', ent_coef=0.1, verbose=1)

model = A2C(CnnPolicy, env, ent_coef=0.1, verbose=1, tensorboard_log=tensorboard_log)

# Train the agent
# Start with 100000.
model.learn(total_timesteps=100000)

# Save the agent
model.save("duckietown-a2c")
