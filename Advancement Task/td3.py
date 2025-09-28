# train via TD3

import env

from stable_baselines3 import TD3
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env


policy_kwargs = dict(net_arch=[256, 256])

env1 = make_vec_env(lambda: env.LegController())

model = TD3("MlpPolicy", env1, verbose=1, device="cuda", policy_kwargs=policy_kwargs)

checkpoint_callback = CheckpointCallback(save_freq=5000, save_path='logs_td3/',
                                         name_prefix='td3')


model.learn(total_timesteps=500000, callback=checkpoint_callback)
