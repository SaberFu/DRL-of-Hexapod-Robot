# train via ppo

import env

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env


policy_kwargs = dict(net_arch=[256, 256])

env1 = make_vec_env(lambda: env.LegController())

model = PPO("MlpPolicy", env1, verbose=1, device="cuda", policy_kwargs=policy_kwargs)

checkpoint_callback = CheckpointCallback(save_freq=5000, save_path='logs_ppo/',
                                         name_prefix='ppo')


model.learn(total_timesteps=500000, callback=checkpoint_callback)