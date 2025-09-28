# conver the command for deployment on robot 

import numpy as np

import env
import time
from stable_baselines3 import DDPG
import math

index = ["011", "012","013","014","015", "016",
         "021", "022","023","024","025", "026",
         "031", "032","033","034","035", "036"]


def convert(action):
    command = np.zeros(18)
    command[0] = -action[5]
    command[1] = -action[4]
    command[2] = -action[3]
    command[3] = -action[0]
    command[4] = action[1]
    command[5] = action[2]
    command[6] = action[14]
    command[7] = action[13]
    command[8] = -action[12]
    command[9] = -action[15]
    command[10] = -action[16]
    command[11] = -action[17]
    command[12] = -action[8]
    command[13] = -action[7]
    command[14] = -action[6]
    command[15] = -action[9]
    command[16] = action[10]
    command[17] = action[11]
    command = [int(rad / math.pi * 2000 + 1500) for rad in command]
    strings = ""
    for i, x in enumerate(command):
        strings += "#" + index[i] + "P{}T0800".format(x) + "!"
    print(strings)
    return command

env2 = env.LegController()

# Load the model
model = DDPG.load("./ddpg/ddpg_380000_steps.zip", device="cpu")

obs = np.zeros(20)

action, _states = model.predict(obs, deterministic=True)


command = convert(action)
return command

