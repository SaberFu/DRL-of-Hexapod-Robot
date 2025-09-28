# visuallization of PPO agent performance

import env
import time
from stable_baselines3 import PPO
import math


env2 = env.LegController()
viewer = env2.robo.my_render()

print(env2.robo.get_euler)

# Load the model
model = PPO.load("./logs_ppo/ppo_300000_steps.zip", device="cpu")

# # Run the trained agent in the environment
obs, _ = env2.reset()
for i in range(100):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, success, info = env2.step(action, viewer)
    # print(math.degrees(env2.robo.get_rotate_angle))
    # print(env2.robo.get_euler)
    # print(env2._get_euler)
    print(action)
    print(obs)

    viewer.sync()
    time.sleep(0.2)

    # if done or success:
    #     # obs, _ = env2.reset()
    #     time.sleep(5)
    #     break
