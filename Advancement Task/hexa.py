# control of robot in mujoco

import math
import numpy as np
import mujoco as mj
import mujoco.viewer

robo_filename = "./scene.xml"


def quat_mul(q1, q2):
    a1, b1, c1, d1 = q1
    a2, b2, c2, d2 = q2

    a = a1 * a2 - b1 * b2 - c1 * c2 - d1 * d2
    b = a1 * b2 + b1 * a2 + c1 * d2 - d1 * c2
    c = a1 * c2 - b1 * d2 + c1 * a2 + d1 * b2
    d = a1 * d2 + b1 * c2 - c1 * b2 + d1 * a2

    return np.array([a, b, c, d])


def quaternion_to_euler(w, x, y, z):
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = math.atan2(t0, t1)

    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch_y = math.asin(t2)

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = math.atan2(t3, t4)

    result = [roll_x, pitch_y, yaw_z]

    return [math.degrees(rad) for rad in result]


class PDController:
    def __init__(self, kp, kd, dt=0.01):
        """
        kp: 比例增益
        kd: 微分增益
        dt: 时间步长（用于离散化微分）
        """
        self.kp = kp
        self.kd = kd
        self.dt = dt
        self.prev_error = 0.0  # 前一时刻的误差

    def compute(self, target, current):
        """
        计算控制输出
        target: 目标值
        current: 当前值
        """
        error = target - current
        # 计算误差变化率（离散化近似）
        error_derivative = (error - self.prev_error) / self.dt
        self.prev_error = error
        # PD控制输出
        output = self.kp * error + self.kd * error_derivative
        return output


class Hexa:
    def __init__(self):
        self.model = mj.MjModel.from_xml_path(filename=robo_filename)
        self.data = mj.MjData(self.model)
        self.pd = [PDController(kp=0.1, kd=0.05) for _ in range(18)]

    def reset(self):
        mj.mj_resetData(self.model, self.data)
        for pdc in self.pd:
            pdc.prev_error = 0

    def step(self, action=None):
        if action is None:
            action = np.zeros(self.model.nu)
        curr_pos = self.get_q
        order = []
        for i, pdc in enumerate(self.pd):
            order.append(pdc.compute(action[i], curr_pos[i]))
        self.data.ctrl = np.array(order)
        mj.mj_step(self.model, self.data)
        return self._get_obs()

    def _get_obs(self):
        state = {
            "q_pos": self.data.qpos.tolist(),
            "ctrl": self.data.ctrl.tolist(),
        }
        return state

    def my_render(self):
        viewer = mj.viewer.launch_passive(self.model, self.data)
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True
        self.model.vis.map.force = 0.03

        return viewer

    def forward(self):
        mujoco.mj_forward(self.model, self.data)

    @property
    def get_pos(self):
        return self.data.qpos[:3]

    @property
    def get_wxyz(self):
        return self.data.qpos[3:7]

    @property
    def get_q(self):
        return self.data.qpos[-18:]

    @property
    def get_euler(self):
        wxyz = self.get_wxyz
        return quaternion_to_euler(*wxyz)

    @property
    def get_rotate_angle(self):
        q = self.get_wxyz
        v_quat = np.array([0, 0, 0, 1])
        q_conj = q * [1, -1, -1, -1]
        tmp = quat_mul(q, v_quat)
        v_world_quat = quat_mul(tmp, q_conj)
        v_world = v_world_quat[1:]
        angle_rad = np.arccos(v_world[2] / np.linalg.norm(v_world))
        return float(angle_rad)
