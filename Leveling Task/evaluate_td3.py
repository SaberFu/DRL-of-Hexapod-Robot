# evaluate TD3 agent performance

import os
import env
import csv

from stable_baselines3 import TD3
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env


def write(a, b):
    with open('output_td3.csv', mode='a', newline='') as file:
        writer = csv.writer(file)

        row = [a, b]

        writer.writerow(row)


def evaluate(model, env, n_eval_episodes):
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=n_eval_episodes)
    write(mean_reward, std_reward)
    print(f"Mean Reward: {mean_reward:.2f} +/- {std_reward:.2f}")


folder = "./logs/"
filenames = os.listdir(folder)

for i, item in enumerate(filenames):
    print(i)
    env_hexa = make_vec_env(lambda: env.LegController())
    name = folder + item
    name = name[:-4]
    model = TD3.load(name, env_hexa)
    evaluate(model, env_hexa, 20)
