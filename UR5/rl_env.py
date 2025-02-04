import time
import math
import random

import numpy as np
import pybullet as p
import pybullet_data

from collections import namedtuple
from attrdict import AttrDict
from tqdm import tqdm
import gymnasium as gym
from gymnasium import spaces

class ReachRL(gym.Env):

    SIMULATION_STEP_DELAY = 1 / 50.

    def __init__(self,robot,camera=None,vis=False) -> None:
        self.robot = robot
        self.vis = vis
        self.camera = camera
        self.step_counter = 0

        self.height = 0.25
        self.x_lower_bound = 0.2
        self.x_higher_bound = 0.35
        self.y_lower_bound = -0.5
        self.y_higher_bound = 0.5
        self.z_lower_bound = 0.15
        self.z_higher_bound = 0.35


        # define environment
        # self.physicsClient = p.connect(p.GUI if self.vis else p.DIRECT)
        # self.planeID = p.loadURDF("plane.urdf")

        # load UR5 and gripper
        # ur5_id = self.robot.load_robot()

        self.low_obs = np.array([-0.5 for i in range(6)])
        self.high_obs = np.array([0.5 for i in range(6)])
        self.action_space = spaces.Box(low=-1,high=1,shape=(6,))
        self.observation_space = spaces.Box(low=self.low_obs,high=self.high_obs)
        
        # target position
        x = np.random.uniform(self.x_lower_bound, self.x_higher_bound)
        y = np.random.uniform(self.y_lower_bound, self.y_higher_bound)
        z = self.height
        self.target_position = np.array([x,y,z])
        self.line_id = p.addUserDebugLine([x,y,z],[x,y,z+0.2],[1,0,0],4)
        (self.ee_pos,self.ee_ori) = self.robot.get_current_pose() 
        self.distance_3d = np.array(self.ee_pos)-self.target_position

    def step_simulation(self):
        """
        Hook p.stepSimulation()
        """
        p.stepSimulation()
        if self.vis:
            time.sleep(self.SIMULATION_STEP_DELAY)

    def step(self, action, control_method='joint'):
        """
        action: (x, y, z, roll, pitch, yaw, gripper_opening_length) for End Effector Position Control
                (a1, a2, a3, a4, a5, a6, a7, gripper_opening_length) for Joint Position Control
        control_method:  'end' for end effector position control
                         'joint' for joint position control
        """
        assert control_method in ('joint', 'end')
        action = [action[i] * math.pi for i in range(6)]
        self.robot.set_joint_angles(action)
        # self.robot.move_gripper(action[-1])

        # self.joint_velocity_pre = self.robot.get_joint_obs()['velocities']

        # for _ in range(5):  # Wait for a few steps
        self.step_simulation()

        self.step_counter += 1
        # self.obs = self.robot.get_joint_obs()
        (self.ee_pos,self.ee_ori) = self.robot.get_current_pose() 
        self.distance_3d = np.array(self.ee_pos)-self.target_position
        # self.joint_velocity = self.robot.get_joint_obs()['velocities']
        # self.joint_acceleration = (np.array(self.joint_velocity) - np.array(self.joint_velocity_pre)) / self.SIMULATION_STEP_DELAY
        reward,done = self.update_reward()
        info = {}
        return np.concatenate((self.ee_pos,self.target_position)), reward, done, False, info
    
    def update_reward(self):
        reward = 0
        done = False
        out = [abs(self.ee_pos[i]) > abs(self.high_obs[i]) for i in range(len(self.ee_pos))]        # print(distance_2d)
        if self.step_counter > 500:
            # reward = -10
            done = True 
            print("Fail!")
        # elif True in out:
        #     reward = -10
        #     done = True
        elif np.linalg.norm(self.distance_3d) <= 0.03:
            reward = 20
            done = True
            print("Reach!")
        else:
            # reward = np.exp(-20*np.linalg.norm(self.distance_3d)) - 0.0001 * np.linalg.norm(self.joint_acceleration)
            # reward = -0.005 * self.distance_3d
            reward = np.exp(-20*np.linalg.norm(self.distance_3d)) - 0.01
        
        return reward,done
    
    def reset(self,seed = None):
        self.robot.reset()
        info = {}
        self.step_counter = 0
        (self.ee_pos,self.ee_ori) = self.robot.get_current_pose()
        # self.joint_velocity = self.robot.get_joint_obs()['velocities']
        # target position
        x = np.random.uniform(self.x_lower_bound, self.x_higher_bound)
        y = np.random.uniform(self.y_lower_bound, self.y_higher_bound)
        z = self.height
        self.target_position = np.array([x,y,z])
        p.removeUserDebugItem(self.line_id)
        self.line_id = p.addUserDebugLine([x,y,z],[x,y,z+0.2],[1,0,0],4)
        # self.distance_3d = (np.array(self.ee_pos)-self.target_position)

        # print(self.distance_3d)
        return np.concatenate((self.ee_pos,self.target_position)),info
    
    def close(self):
        p.disconnect(self.physicsClient)

# if __name__ == "__main__":
#     robot = MagneticReach(pos=(0,0.5,0),ori=(0,0,0))
#     robot.load()