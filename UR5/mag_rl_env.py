import time
import math
import numpy as np
import pybullet as p
import gymnasium as gym
from gymnasium import spaces
from MagneticEngine import permanent_magnet

class MagneticReachRL(gym.Env):
    SIMULATION_STEP_DELAY = 1 / 100.

    def __init__(self,robot) -> None:
        self.robot = robot
        self.vis = robot.vis

        # 设置不同奖励项的比例系数
        self.k_balance = 20
        self.k_target = 20

        # 设置机械臂末端朝向为竖直向下
        self.ee_euler = [math.pi/2,math.pi/2,-math.pi/2]

        # 设置单个回合最大步数
        self.max_step_count = 1000

        # 设置磁矩大小
        self.moment_source = np.array([0.0,0,26.2]).reshape(3,1)
        self.moment_capsule = np.array([0.0,0,-0.126]).reshape(3,1)

        # 浮力，重力，平衡时的距离
        self.buoyant_force = [0,0,0.0148]
        self.balance_dis = 0.25
        self.gravity = [0,0,-0.0153]

        # 目标位置的范围
        self.x_lower_bound = 0.3
        self.x_higher_bound = 0.5
        self.y_lower_bound = 0.1
        self.y_higher_bound = 0.2
        self.z_lower_bound = 0.1
        self.z_higher_bound = 0.2
        self.point_id = p.addUserDebugPoints([[0.0,0.0,0.0]],[[1,0,0]],pointSize=1)

        # 观察空间的范围
        self.low_obs = np.array([-1 for i in range(4)])
        self.high_obs = np.array([1 for i in range(4)])

        # 动作空间，观察空间
        self.action_space = spaces.Discrete(2)
        # self.action_space = spaces.Box(low=-1,high=1,shape=(1,))
        self.observation_space = spaces.Box(low=self.low_obs,high=self.high_obs)
        self.action_to_direction = {0:[0.0,0,-0.01],1:[0,0,0.01]}

        self.reset()

    def reset(self,seed = None):
        self.robot.reset()
        ee_pos = self.robot.get_current_pose()[0]
        self.robot.reset_magnet(ee_pos)
        info = {}
        self.step_counter = 0
        
        # 设定随机目标位置
        x = np.random.uniform(self.x_lower_bound, self.x_higher_bound)
        y = np.random.uniform(self.y_lower_bound, self.y_higher_bound)
        z = np.random.uniform(self.z_lower_bound, self.z_higher_bound)

        self.target_position = np.array([ee_pos[0],ee_pos[1],z])
        self.capsule_position,self.capsule_orientation = p.getBasePositionAndOrientation(self.robot.capsule)

        p.removeUserDebugItem(self.point_id)
        self.point_id = p.addUserDebugPoints([self.target_position],[[1,0,0]],pointSize=10)

        # 平衡距离
        balance_dis = np.array([ee_pos[2]-self.capsule_position[2]])
        self.step_simulation()

        return np.concatenate((balance_dis,self.target_position)),info
    
    def step(self,action):
        ee_pos = self.robot.get_current_pose()[0]

        # 采取action后到达的位置
        position = self.action_to_direction[int(action)] + np.array([round(ee_pos[i],2)for i in range(len(ee_pos))])

        # 逆运动学计算，设定关节角
        joint_angles = self.robot.calculate_ik(position,self.ee_euler)
        self.robot.set_joint_angles(joint_angles)

        self.step_counter += 1

        # 执行一步仿真
        self.step_simulation()

        # 获取机械臂下一时刻的末端位置
        ee_pos = self.robot.get_current_pose()[0]

        ee_pos_trans = np.array(ee_pos).reshape(3,1) 
        capsule_position_trans = np.array(self.capsule_position).reshape(3,1)

        # 计算磁场力和磁场力矩
        magnetic_force,magnetic_torque = permanent_magnet.force_moment(ee_pos_trans-capsule_position_trans,
                                                                      self.moment_source,self.moment_capsule)

        magnetic_force_trans = [magnetic_force.T[0,i] for i in range(3)]

        # 施加外力
        p.applyExternalForce(self.robot.capsule, -1, magnetic_force_trans, self.capsule_position,p.WORLD_FRAME)
        p.applyExternalForce(self.robot.capsule,-1,self.buoyant_force,self.capsule_position,p.WORLD_FRAME)
        p.applyExternalForce(self.robot.capsule,-1,self.gravity,self.capsule_position,p.WORLD_FRAME)

        # 执行一步仿真
        self.step_simulation()

        # 获取机械臂下一时刻的末端位置，获取胶囊下一时刻的位置
        ee_pos = self.robot.get_current_pose()[0]
        self.capsule_position,self.capsule_orientation = p.getBasePositionAndOrientation(self.robot.capsule)

        # 计算奖励
        reward,done = self.update_reward(ee_pos)
        info = {}
        balance_dis = np.array([ee_pos[2]-self.capsule_position[2]])

        return np.concatenate((balance_dis,self.target_position)),reward, done, False, info

    def update_reward(self,ee_pos):
        reward = 0
        done = False
        if self.step_counter > self.max_step_count:
            done = True 
            print("Fail!")
        else:
            reward = np.exp(-self.k_balance*np.linalg.norm(ee_pos[2]-self.capsule_position[2]-self.balance_dis))
            + np.exp(-self.k_target*np.linalg.norm(self.capsule_position[2]-self.target_position[2]))

        return reward,done

    def step_simulation(self):
        p.stepSimulation()
        if self.vis:
            time.sleep(self.SIMULATION_STEP_DELAY)

    def close(self):
        p.disconnect(self.physicsClient)

    