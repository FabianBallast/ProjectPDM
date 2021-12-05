# from typing_extensions import ParamSpec
import numpy as np
from numpy.core.shape_base import hstack
from scipy.spatial.transform import Rotation, rotation

class controlller:
    def __init__(self):
        # hover control gains
        self.Kp = np.diag([40, 40, 40])
        self.Kd = np.diag([9, 9, 9])
        # angular control gains
        self.K_R = np.diag([2500, 2500, 400])
        self.K_w = np.diag([60, 60, 50])
        m = 0.030  # weight (in kg) with 5 vicon markers (each is about 0.25g)
        g = 9.81  # gravitational constant
        I = np.array([[1.43e-5, 0, 0],
                      [0, 1.43e-5, 0],
                      [0, 0, 2.89e-5]])  # inertial tensor in m^2 kg
        L = 0.046  # arm length in m
        self.rotor_speed_min = 0
        self.rotor_speed_max = 2500
        self.mass = m
        self.inertia = I
        self.invI = np.linalg.inv(I)
        self.g = g
        self.arm_length = L
        self.k_thrust = 2.3e-08
        self.k_drag = 7.8e-11

    def control(self, flat_output, state):
        '''
        :param desired state: pos, vel, acc, yaw, yaw_dot
        :param current state: pos, vel, euler, omega
        :return:
        '''
        cmd_motor_speeds = np.zeros((4,))
        cmd_thrust = 0
        cmd_moment = np.zeros((3,))
        cmd_q = np.zeros((4,))

        error_pos = flat_output[0] - state['x']
        error_vel = flat_output[1] - state['v']

        # page 29
        rdd_des = flat_output[2].T + self.Kd @ error_vel.T + self.Kp @ error_pos.T
    
        # Desired roll, pitch and yaw (in rad).
        # page 30 Equation 8
        

        # calculate u1 (thrust)
        # page 30 Equation 8
        F_des = (self.mass * (rdd_des + np.array([0,0,self.g]))).reshape((3,1))
        R = Rotation.as_matrix(Rotation.from_quat(state['q']))
        b3 = R[:, 2:3]
        u1 = b3.T @ F_des

        # print(f"Target angles {np.array([phi_des, theta_des, psi_des])}")
        # page 31
        u2 = 0
        if np.linalg.norm(F_des > 0.00001):
            b3_des = F_des / np.linalg.norm(F_des)
            a_Psi = np.array([np.cos(flat_output[3]), np.sin(flat_output[3]), 0]).reshape((3,1))
            b2_des = np.cross(b3_des, a_Psi, axis=0) / np.linalg.norm(np.cross(b3_des, a_Psi, axis=0))
            b1_des = np.cross(b2_des, b3_des, axis=0)
            # print(a_Psi)
            R_des = np.hstack((b1_des, b2_des, b3_des))
            # print(R_des)
            temp = R_des.T @ R -R.T @ R_des
            R_temp = 0.5 * temp
            e_R = 0.5 * np.array([-R_temp[1,2], R_temp[0,2], -R_temp[0,1]])
            u2 = self.inertia @ (-self.K_R @ e_R -self.K_w @ (np.array(state['w'])))
        else:
            u2 = np.array([0,0,0])


        gama = self.k_drag / self.k_thrust
        Len = self.arm_length
        cof_temp = np.array(
            [1, 1, 1, 1, 0, Len, 0, -Len, -Len, 0, Len, 0, gama, -gama, gama, -gama]).reshape(4, 4)
   
        u = np.array([u1[0][0], u2[0], u2[1], u2[2]])
        F_i = np.matmul(np.linalg.inv(cof_temp), u)
     
        for i in range(4):
            if F_i[i] < 0:
                F_i[i] = 0
                cmd_motor_speeds[i] = self.rotor_speed_min
            cmd_motor_speeds[i] = np.sqrt(F_i[i] / self.k_thrust)
            if cmd_motor_speeds[i] > self.rotor_speed_max:
                cmd_motor_speeds[i] = self.rotor_speed_max


        cmd_thrust = u1
        cmd_moment[0] = u2[0]
        cmd_moment[1] = u2[1]
        cmd_moment[2] = u2[2]


        control_input = {'cmd_motor_speeds': cmd_motor_speeds,
                         'cmd_thrust': cmd_thrust,
                         'cmd_moment': cmd_moment}
        return control_input
