import os
import numpy as np
from physical_env import PhysicalEnv
import math
from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from rl_env import ReachRL
import argparse
from mag_rl_env import MagneticReachRL

# 选择仿真类型
class choose_simulation_type():
    def __init__(self,args):
        self.args = args

    # 逆运动学仿真
    def demo_slider_simulation(self):
        """ Demo program showing how to use the sim """
        env = PhysicalEnv(vis=self.args.render)
        env.add_gui_sliders()
        while True:
            x, y, z, Rx, Ry, Rz = env.read_gui_sliders()
            joint_angles = env.calculate_ik([x, y, z], [Rx, Ry, Rz])
            # print(joint_angles)
            env.draw_debug_line(x,y,z)
            env.set_joint_angles(joint_angles)
            env.check_collisions()
            env.step_simulation()
    
    # 逆运动学画圆
    def demo_circle_simulation(self):
        env = PhysicalEnv(vis=self.args.render)
        time = 0
        [Rx, Ry, Rz] = [math.pi/2,math.pi/2,-math.pi/2]
        while True:
            time += env.time_step
            [x,y,z] = [0.2 + 0.05*math.sin(time),0.0+0.05*math.cos(time),0.5]
            joint_angles = env.calculate_ik([x, y, z], [Rx, Ry, Rz])
        
            env.draw_debug_line(x,y,z)

            env.set_joint_angles(joint_angles)
            env.check_collisions()
            env.step_simulation()
    
    # 关节空间的强化学习
    def demo_joint_space_rl(self):
        model_path = 'model/rl_reach.zip'

        robot = PhysicalEnv(vis=self.args.render)
        env = ReachRL(robot,camera=None,vis=self.args.render)
        if self.args.test == True:
            model = PPO.load(model_path)
            obs,_ = env.reset()
            done = False
            while not done:
                 action,states = model.predict(obs)
                 obs, reward, done, _, info = env.step(action)
        else:
            env.reset()
            if os.path.exists(model_path):
                print('continue!')
                model = PPO.load(model_path, env=env,device='cpu')
                model.learn(total_timesteps=500000)

            else:
                print('start!')
                model = PPO("MlpPolicy", env, verbose=1,device='cpu')
                model.learn(total_timesteps=250000)
                            
            model.save(model_path)

    # 磁控强化学习-逆运动学
    def demo_magnetic_rl(self):
        model_path = 'model/mag_rl_reach.zip'

        robot = PhysicalEnv(vis=self.args.render)
        robot.load_magnet()
        env = MagneticReachRL(robot)
        if self.args.test == True:
            model = PPO.load(model_path)
            obs,_ = env.reset()
            done = False
            while not done:
                 action,states = model.predict(obs)
                 obs, reward, done, _, info = env.step(action)
        else:
            env.reset()
            if os.path.exists(model_path):
                print('continue!')
                model = PPO.load(model_path, env=env,device='cpu')
                model.learn(total_timesteps=1000000)

            else:
                print('start!')
                model = PPO("MlpPolicy", env, verbose=1,device='cpu')
                model.learn(total_timesteps=250000)
                
            model.save(model_path)

# 命令行传递参数
def parse_arguments():
    parse = argparse.ArgumentParser(description="parse arguments")
    parse.add_argument('--test',action="store_true",help='test')
    # parse.add_argument('--train',action="store_false",help='train')
    parse.add_argument('--render',action="store_true",help='render')
    # parse.add_argument('--no_render',action="store_false",help='do not render')
    args = parse.parse_args()
    return args

        
if __name__ == "__main__":
    args = parse_arguments()
    sim = choose_simulation_type(args)
    # sim.demo_circle_simulation()
    # sim.demo_slider_simulation()
    # sim.demo_joint_space_rl()
    sim.demo_magnetic_rl()