import gym

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2

import utils
tensorboard_log = "logs/cartpole-ppo2"
utils.mkdir_p(tensorboard_log)

env = gym.make('CartPole-v1')
# Vectorized environments allow to easily multiprocess training
# we demonstrate its usefulness in the next examples
env = DummyVecEnv([lambda: env])  # The algorithms require a vectorized environment to run

model = PPO2(MlpPolicy, env, verbose=1, tensorboard_log=tensorboard_log)
# Train the agent
model.learn(total_timesteps=1000000)

# Save the agent
model.save("cartpole-ppo2")
