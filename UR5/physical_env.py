import os
import math 
import pybullet 
import pybullet_data
from collections import namedtuple
from attrdict import AttrDict

ROBOT_URDF_PATH = os.path.join(os.getcwd(),"/ur_e_description/urdf/ur5e.urdf")
TABLE_URDF_PATH = os.path.join(pybullet_data.getDataPath(), "table/table.urdf")
PLANE_URDF_PATH = os.path.join(pybullet_data.getDataPath(), "plane.urdf")
SOURCE_URDF_PATH = os.path.join(os.getcwd(),"/ur_e_description/urdf/permanent_magnet.urdf")
CAPSULE_URDF_PATH = os.path.join(os.getcwd(),"/ur_e_description/urdf/sphere_small.urdf")

class PhysicalEnv():
  
    def __init__(self, vis=False):
        self.vis = vis
        pybullet.connect(pybullet.GUI if self.vis else pybullet.DIRECT)
        
        # 平衡距离
        self.balance_dis = 0.25

        # 机械臂初始位姿,注意，0处为基座，不影响机械臂姿态
        self.rest_poses = [0, math.pi/6, -math.pi/2, math.pi/2, math.pi/2, math.pi/2]

        # 设置仿真步长
        self.time_step = 1./100.
        pybullet.setTimeStep(self.time_step)
        
        self.end_effector_index = 7
        self.ur5 = self.load_robot()
        self.num_joints = pybullet.getNumJoints(self.ur5)
                
        self.control_joints = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]
        self.joint_type_list = ["REVOLUTE", "PRISMATIC", "SPHERICAL", "PLANAR", "FIXED"]
        self.joint_info = namedtuple("jointInfo", ["id", "name", "type", "lowerLimit", "upperLimit", "maxForce", "maxVelocity", "controllable"])

        self.joints = AttrDict()
        for i in range(self.num_joints):
            info = pybullet.getJointInfo(self.ur5, i)
            jointID = info[0]
            jointName = info[1].decode("utf-8")
            jointType = self.joint_type_list[info[2]]
            jointLowerLimit = info[8]
            jointUpperLimit = info[9]
            jointMaxForce = info[10]
            jointMaxVelocity = info[11]
            controllable = True if jointName in self.control_joints else False
            info = self.joint_info(jointID, jointName, jointType, jointLowerLimit, jointUpperLimit, jointMaxForce, jointMaxVelocity, controllable)
            if info.type == "REVOLUTE":
                pybullet.setJointMotorControl2(self.ur5, info.id, pybullet.VELOCITY_CONTROL, targetVelocity=0, force=0)
            self.joints[info.name] = info     

    # 加载模型
    def load_robot(self):
        flags = pybullet.URDF_USE_SELF_COLLISION
        plane = pybullet.loadURDF(PLANE_URDF_PATH)
        robot = pybullet.loadURDF(ROBOT_URDF_PATH, flags=flags)
        return robot
        
    # 设置关节角度
    def set_joint_angles(self, joint_angles):
        poses = []
        indexes = []
        forces = []

        for i, name in enumerate(self.control_joints):
            joint = self.joints[name]
            poses.append(joint_angles[i])
            indexes.append(joint.id)
            forces.append(joint.maxForce)

        pybullet.setJointMotorControlArray(
            self.ur5, indexes,
            pybullet.POSITION_CONTROL,
            targetPositions=joint_angles,
            targetVelocities=[0]*len(poses),
            positionGains=[0.04]*len(poses), forces=forces
        )

    # 获取关节角度
    def get_joint_angles(self):
        j = pybullet.getJointStates(self.ur5, [1,2,3,4,5,6])
        joints = [i[0] for i in j]
        return joints
    
    # 逆运动学计算
    def calculate_ik(self, position, orientation):
        quaternion = pybullet.getQuaternionFromEuler(orientation)
        lower_limits = [-math.pi]*6
        upper_limits = [math.pi]*6
        joint_ranges = [2*math.pi]*6
        joint_angles = pybullet.calculateInverseKinematics(
            self.ur5, self.end_effector_index, position, quaternion, 
            jointDamping=[0.01]*6, upperLimits=upper_limits, 
            lowerLimits=lower_limits, jointRanges=joint_ranges, 
            restPoses=self.rest_poses
        )

        return joint_angles
    
    # 加载GUI滑块
    def add_gui_sliders(self):
        self.sliders = []
        self.sliders.append(pybullet.addUserDebugParameter("X", -0.5, 0.5, 0.2))
        self.sliders.append(pybullet.addUserDebugParameter("Y", -1, 1, 0.0))
        self.sliders.append(pybullet.addUserDebugParameter("Z", 0.3, 1, 0.5))
        self.sliders.append(pybullet.addUserDebugParameter("Rx", -math.pi/2, math.pi/2, math.pi/2))
        self.sliders.append(pybullet.addUserDebugParameter("Ry", -math.pi/2, math.pi, math.pi/2))
        self.sliders.append(pybullet.addUserDebugParameter("Rz", -math.pi/2, math.pi/2, -math.pi/2))

    # 读取GUI滑块
    def read_gui_sliders(self):
        x = pybullet.readUserDebugParameter(self.sliders[0])
        y = pybullet.readUserDebugParameter(self.sliders[1])
        z = pybullet.readUserDebugParameter(self.sliders[2])
        Rx = pybullet.readUserDebugParameter(self.sliders[3])
        Ry = pybullet.readUserDebugParameter(self.sliders[4])
        Rz = pybullet.readUserDebugParameter(self.sliders[5])
        return [x, y, z, Rx, Ry, Rz]
    
    # 获取机械臂末端位置和姿态
    def get_current_pose(self):
        linkstate = pybullet.getLinkState(self.ur5, self.end_effector_index, computeForwardKinematics=True)
        position, orientation = linkstate[0], linkstate[1]
        return (position, orientation)
    
    # 辅助线
    def draw_debug_line(self,x,y,z):
        line_start_x = [x-0.2,y,z]
        line_end_x = [x+0.2,y,z]
        # line_id_x = pybullet.addUserDebugLine(line_start_x,line_end_x,[1,0,0],1)

        line_start_z = [x,y,z+0.2]
        line_end_z = [x,y,z-0.2]
        line_id_z = pybullet.addUserDebugLine(line_start_z,line_end_z,[1,0,0],1)
        # pybullet.removeUserDebugItem(line_id_x)
        # pybullet.removeUserDebugItem(line_id_z)

    # 重置机械臂姿态
    def reset(self):
        for i in range(len(self.rest_poses)):
            pybullet.resetJointState(self.ur5, i,self.rest_poses[i])

    # 重置磁器件
    def reset_magnet(self,ee_pos):
        pybullet.resetBasePositionAndOrientation(self.capsule,[ee_pos[0],ee_pos[1],ee_pos[2]-self.balance_dis],[0,0,0,1])

    # 加载磁器件
    def load_magnet(self):
        self.capsule = pybullet.loadURDF(CAPSULE_URDF_PATH)