# visuallization of TD3 agent performance
import env
import time
from stable_baselines3 import TD3
import math


env2 = env.LegController()
viewer = env2.robo.my_render()

# Load the model
model = TD3.load("./td3/td3_500000_steps.zip", device="cpu")

# # Run the trained agent in the environment
obs, _ = env2.reset()
for i in range(30):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, _, info = env2.step(action)
    print(math.degrees(env2.robo.get_rotate_angle))
    # print(env2._get_euler)
    # print(action)
    # print(obs)

    viewer.sync()
    time.sleep(0.4)
    if done:
        obs, _ = env2.reset()
