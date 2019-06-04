import gym

from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.deepq.policies import MlpPolicy
from stable_baselines import DQN


# Create and wrap the environment.
env = gym.make('LunarLander-v2')
env = DummyVecEnv([lambda: env])

# Create the agent.
tensorboard_log = "logs/lunarlander-dqn"
model = DQN(MlpPolicy, env, verbose=1, tensorboard_log=tensorboard_log)

# Train the agent
model.learn(total_timesteps=1000000)

# Save the agent
model.save("lunarlander-dqn")
