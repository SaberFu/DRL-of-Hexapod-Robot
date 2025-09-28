# Training Env of Leveling Task

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import math
import hexa

class LegController(gym.Env):
    def __init__(self):
        super(LegController, self).__init__()

        self._ctrl_cost_weight = 600
        self._goal_weight = 9000

        self.robo = hexa.Hexa()
        self.reward = 0.0
        self.action = np.zeros(18)
        self.obs = np.zeros(20)
        self.init_angle = 0
        self.angle = 0


        high = np.ones(18) * math.pi / 6
        high = high.astype(np.float32)
        low = -high

        high2 = np.ones(20) * math.pi / 6
        high2 = high2.astype(np.float32)
        high2[-1] = 90
        high2[-2] = 90
        low2 = -high2

        # Define Observation and Action Spaces
        self.action_space = spaces.Box(low, high)
        self.observation_space = spaces.Box(low2, high2)

        # Reset state and time
        self.reset()

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
        self.reward = 0
        self.angle = 0
        self.action = np.zeros(18)
        self.obs = np.zeros(20)
        self.robo.reset()

        # random init of slope
        angle = np.random.uniform(-math.pi / 12, math.pi / 12)
        while abs(angle) < math.radians(5):
            angle = np.random.uniform(-math.pi / 12, math.pi / 12)
        self.init_angle = angle
        angle2 = np.random.uniform(0, math.pi)
        self.robo.model.geom_quat[0] = (
            np.array([math.cos(angle / 2),
                      math.sin(angle / 2) * math.cos(angle2),
                      math.sin(angle / 2) * math.sin(angle2),
                      0]))

        for _ in range(400):
            self.robo.step(np.zeros(18))

        self.obs = self._get_obs()
        self.angle = self.robo.get_rotate_angle

        return self._get_obs(), {}

    def _get_obs(self):
        euler = self.robo.get_euler
        q = self.robo.get_q

        state = []
        for x in q:
            state.append(x)
        for i in range(2):
            state.append(euler[i])
        return np.array(state, dtype=np.float32)

    def _get_rew(self):
        reward = self.angle - self.robo.get_rotate_angle
        rewards = reward * self._goal_weight
        costs = self.control_cost()
        reward = rewards - costs
        reward_info = {
            "reward_ctrl": -costs,
            "reward_track": rewards,
        }
        return reward, reward_info

    def control_cost(self):
        return self._ctrl_cost_weight * float(np.linalg.norm(self.action-self.obs[:18]))

    def step(self, action):
        self.action = action

        for _ in range(100):
            self.robo.step(action)

        reward, reward_info = self._get_rew()

        obs = self._get_obs()
        self.obs = obs
        self.angle = self.robo.get_rotate_angle
        self.reward += reward

        success_goal = False
        if abs(self.angle) < math.radians(3):
            success_goal = True
            print(1,self.reward)

        stop = False
        if self.robo.data.time > 10:
            stop = True


        return obs, reward, stop, success_goal, reward_info