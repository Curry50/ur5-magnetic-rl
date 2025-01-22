import os
import numpy as np
from datetime import datetime
from collections import namedtuple
from attrdict import AttrDict
from IK import UR5sim
import time
import math
from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import PPO
from rl_env import MagneticReach
import argparse

class choose_simulation_type():
    def __init__(self):
        pass

    def demo_slider_simulation(self):
        """ Demo program showing how to use the sim """
        env = UR5sim()
        env.add_gui_sliders()
        while True:
            x, y, z, Rx, Ry, Rz = env.read_gui_sliders()
            joint_angles = env.calculate_ik([x, y, z], [Rx, Ry, Rz])
            # print(joint_angles)
            env.draw_debug_line(x,y,z)
            env.set_joint_angles(joint_angles)
            env.check_collisions()
    
    def demo_circle_simulation(self):
        env = UR5sim()
        time = 0
        [Rx, Ry, Rz] = [math.pi/2,math.pi/2,-math.pi/2]
        while True:
            time += env.time_step
            [x,y,z] = [0.2 + 0.05*math.sin(time),0.0+0.05*math.cos(time),0.5]
            joint_angles = env.calculate_ik([x, y, z], [Rx, Ry, Rz])
        
            env.draw_debug_line(x,y,z)

            env.set_joint_angles(joint_angles)
            env.check_collisions()
        
    def demo_joint_space_rl(self,test=False,visual=False):
        model_path = 'magnetic_reach.zip'

        robot = UR5sim(vis=visual)
        env = MagneticReach(robot,camera=None,vis=visual)
        if test == True:
            model = PPO.load(model_path)
            obs,_ = env.reset()
            done = False
            while not done:
                 action,states = model.predict(obs)
                 obs, reward, done, _, info = env.step(action)
                 print('distance:',env.distance_3d)
        else:
            env.reset()
            if os.path.exists(model_path):
                print('continue!')
                model = PPO.load(model_path, env=env,device='cpu')
                model.learn(total_timesteps=250000)

            else:
                print('start!')
                model = PPO("MlpPolicy", env, verbose=1,device='cpu')
                model.learn(total_timesteps=250000)
                
            
            model.save(model_path)
    
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
    sim = choose_simulation_type()
    # sim.demo_circle_simulation()
    # sim.demo_slider_simulation()
    sim.demo_joint_space_rl(args.test,args.render)